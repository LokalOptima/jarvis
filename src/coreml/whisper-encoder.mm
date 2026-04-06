#import "whisper-encoder.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <cstring>

@interface WhisperEncoderModel : NSObject
@property (nonatomic, strong) MLModel *model;
- (instancetype)initWithPath:(NSString *)path;
- (BOOL)encodeWithMel:(float *)mel nCtx:(int64_t)nCtx nMel:(int64_t)nMel out:(float *)out;
@end

@implementation WhisperEncoderModel

- (instancetype)initWithPath:(NSString *)path {
    self = [super init];
    if (self) {
        NSURL *url = [NSURL fileURLWithPath:path];
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;

        NSError *error = nil;

        // Try loading as pre-compiled .mlmodelc first.
        _model = [MLModel modelWithContentsOfURL:url configuration:config error:&error];

        // If that fails, try compiling from .mlpackage at runtime.
        if (!_model) {
            error = nil;
            NSURL *compiledURL = [MLModel compileModelAtURL:url error:&error];
            if (error || !compiledURL) {
                NSLog(@"Failed to compile Core ML model: %@", error);
                return nil;
            }
            _model = [MLModel modelWithContentsOfURL:compiledURL configuration:config error:&error];
            if (error || !_model) {
                NSLog(@"Failed to load compiled Core ML model: %@", error);
                return nil;
            }
        }
    }
    return self;
}

- (BOOL)encodeWithMel:(float *)mel nCtx:(int64_t)nCtx nMel:(int64_t)nMel out:(float *)out {
    NSError *error = nil;

    MLMultiArray *inputArray = [[MLMultiArray alloc]
        initWithDataPointer:mel
                      shape:@[@1, @(nMel), @(nCtx)]
                   dataType:MLMultiArrayDataTypeFloat32
                    strides:@[@(nCtx * nMel), @(nCtx), @1]
                deallocator:nil
                      error:&error];
    if (error) {
        NSLog(@"Failed to create input MLMultiArray: %@", error);
        return NO;
    }

    MLDictionaryFeatureProvider *input =
        [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{@"logmel_data": inputArray}
                         error:&error];
    if (error) {
        NSLog(@"Failed to create feature provider: %@", error);
        return NO;
    }

    id<MLFeatureProvider> output = [_model predictionFromFeatures:input error:&error];
    if (error) {
        NSLog(@"Core ML prediction failed: %@", error);
        return NO;
    }

    MLMultiArray *result = [output featureValueForName:@"output"].multiArrayValue;
    if (!result) return NO;

    // MLMultiArray may have non-contiguous strides (ANE padding).
    // Copy element-by-element using getBytesWithHandler for correct layout.
    [result getBytesWithHandler:^(const void *bytes, NSInteger size) {
        // If the backing buffer is exactly the expected size, it's contiguous.
        NSInteger expected = result.count * sizeof(float);
        if (size == expected) {
            memcpy(out, bytes, size);
        } else {
            // Non-contiguous: fall back to indexed access.
            for (NSInteger i = 0; i < result.count; i++) {
                out[i] = [[result objectAtIndexedSubscript:i] floatValue];
            }
        }
    }];

    return YES;
}

@end

struct whisper_coreml_context {
    void * encoder;  // WhisperEncoderModel *, prevent ARC bridging issues
};

struct whisper_coreml_context * whisper_coreml_init(const char * path_model) {
    NSString *path = [NSString stringWithUTF8String:path_model];

    WhisperEncoderModel *encoder = [[WhisperEncoderModel alloc] initWithPath:path];
    if (!encoder) {
        return nullptr;
    }

    auto *ctx = new whisper_coreml_context;
    ctx->encoder = (__bridge_retained void *)encoder;
    return ctx;
}

void whisper_coreml_free(struct whisper_coreml_context * ctx) {
    if (ctx) {
        CFBridgingRelease(ctx->encoder);
        delete ctx;
    }
}

void whisper_coreml_encode(
        const struct whisper_coreml_context * ctx,
        int64_t n_ctx,
        int64_t n_mel,
        float * mel,
        float * out) {
    @autoreleasepool {
        WhisperEncoderModel *encoder = (__bridge WhisperEncoderModel *)ctx->encoder;
        [encoder encodeWithMel:mel nCtx:n_ctx nMel:n_mel out:out];
    }
}
