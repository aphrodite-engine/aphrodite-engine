import time
import wave

from aphrodite.modeling.models.orpheus import OrpheusModel


def main():
    model = OrpheusModel(
        model_name="canopylabs/orpheus-tts-0.1-finetune-prod",
        max_model_len=2048,
        gpu_memory_utilization=0.1,
        )
    prompt = r"""Man, the way social media has, um, completely changed how
                 we interact is just wild, right? Like, we're all connected 24/7
                 but somehow people feel more alone than ever. And don't even get
                 me started on how it's messing with kids' self-esteem and mental
                 health and whatnot."""

    start_time = time.monotonic()
    syn_tokens = model.generate_speech(
       prompt=prompt,
       voice="tara",
       )

    with wave.open("output.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        total_frames = 0
        chunk_counter = 0
        for audio_chunk in syn_tokens: # output streaming
            chunk_counter += 1
            frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
            total_frames += frame_count
            wf.writeframes(audio_chunk)
        duration = total_frames / wf.getframerate()

        end_time = time.monotonic()
        print(
            f"It took {end_time - start_time} seconds to generate {duration:.2f} "
            "seconds of audio")


if __name__ == '__main__':
    main()
