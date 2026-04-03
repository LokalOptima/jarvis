# Building a Personalized Wake Word Detector with Whisper

I wanted a wake word detector for my family's home assistant — something that responds to "hey Jarvis" spoken by specific people, runs on CPU, needs no cloud, and doesn't require training a neural network from scratch.

## The Problem with Off-the-Shelf Solutions

Existing wake word systems like openWakeWord detect generic keywords ("hey Mycroft", "Alexa") but can't be personalized to a specific voice. They're trained on large datasets and frozen — you get what you get.

I needed something that could be enrolled with just a handful of recordings from each family member.

## Architecture: Whisper as a Feature Extractor

The key insight: use a frozen speech encoder as a feature extractor, then do template matching on the output. No training required.

**Whisper Tiny** (8.2M parameters, 384-dim embeddings) turns audio into a sequence of feature vectors — one per ~20ms of audio. These vectors capture phonetic content, speaker characteristics, and temporal structure. Two utterances of "hey Jarvis" by the same speaker produce similar feature sequences; random speech produces different ones.

The pipeline:

```
Mic audio (16kHz) → Whisper mel spectrogram → Whisper encoder → [T, 384] features
                                                                      ↓
                                                              Subsequence DTW
                                                              against templates
                                                                      ↓
                                                              Score > threshold?
                                                                  → DETECTED
```

## Enrollment

Enrollment is a two-step process:

**Step 1: Record.** Say "hey Jarvis" repeatedly into the mic. Whisper Turbo (a much larger model) runs in the background with word-level timestamps to automatically find each occurrence in the recording. Each detection is extracted as a trimmed audio clip.

An important design decision: the live Whisper feedback during recording is approximate — it processes 3-second chunks and might miss wake words that straddle chunk boundaries. The real extraction runs on the complete recording after you stop. This avoids the boundary problem without complicating the streaming logic.

**Step 2: Review and build.** A web UI lets you listen to each clip and delete bad ones. Then `make build` runs each clip through Whisper Tiny's encoder and saves the full frame sequences as templates.

## The C++ Runtime

The always-on detector is a compiled C++ binary using a vendored, stripped-down copy of whisper.cpp (2,800 lines, down from 7,400 — decoder, tokenizer, grammar, and GPU backends all removed).

Every 200ms it:

1. Grabs the last 2 seconds from an SDL2 audio ring buffer
2. Checks energy — if silent, skip
3. Computes the mel spectrogram and runs the Whisper encoder
4. Matches against enrolled templates using subsequence DTW
5. If the score exceeds the threshold, fire a detection

### Variable-Length Encoding

Whisper.cpp was designed for 30-second transcription chunks. It pads all audio to 30 seconds and always encodes 1500 frames. For a 2-second wake word window, that's 93% wasted computation.

The fix: whisper.cpp already had an internal `exp_n_audio_ctx` field for overriding the frame count, used by all three graph builders (convolution, encoder, cross-attention). We just exposed it as a public API:

```c
void whisper_set_audio_ctx(struct whisper_context * ctx, int n_audio_ctx);
```

After computing the mel spectrogram, we calculate the actual number of encoder frames needed (`ceil(mel_frames / 2)`, since the conv has stride 2) and set it before encoding. Result: encoding ~100 frames instead of 1500, roughly 15x less work.

We also removed the 30-second zero-padding from the mel spectrogram computation. The mel now only produces frames for the actual audio.

## Template Matching with Subsequence DTW

The core matching algorithm is subsequence DTW (Dynamic Time Warping). Unlike full DTW, which aligns two complete sequences, subsequence DTW finds the best-matching *region* within a longer input for a shorter template. The template must be fully consumed, but can match anywhere in the input.

The cost function is `1 - cosine_similarity` between frame pairs. Templates are L2-normalized at enrollment time, so the runtime only needs to compute the input frame norms (once per cycle, not per template).

Implementation uses a two-row sliding window instead of a full cost matrix — O(n*m) time, O(m) space.

## What Didn't Work

**Average pooling.** The first attempt collapsed the entire `[T, 384]` frame sequence into a single `[384]` vector by averaging, then compared with cosine similarity. Random noise scored 0.85+ against templates — there was almost no discrimination. Averaging destroys the temporal pattern that makes "hey Jarvis" distinct from other speech.

**Raw DTW without normalization.** Even with proper frame-level DTW, the gap between positive (0.85) and negative (0.63) scores was only 0.22. Workable but fragile. The next post covers what we did about that.
