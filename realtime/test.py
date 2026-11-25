
from obspy import read

from config import DETECTION_SETTINGS
from detection import process_detection
from preprocess import prepare_stream, prepare_components


FILENAME = "NP_EQM07_20261106_0100_0200.mseed"

STATION_ID = "NP.EQM07"


def main():

    print("PREPROCESSING + DETECTION TEST")

    # Step 1: Read MiniSEED file.
    stream = read(FILENAME)

    print("\nORIGINAL STREAM")
    print(stream)

    # Step 2: Preprocess waveform.
    processed = prepare_stream(
        stream,
        fs=DETECTION_SETTINGS["fs"],
        fmin=DETECTION_SETTINGS["fmin"],
        fmax=DETECTION_SETTINGS["fmax"],
    )

    print("\nPROCESSED STREAM")
    print(processed)

    # Step 3: Prepare vertical and horizontal components.
    components = prepare_components(processed)

    # Step 4: Test P-wave detection using vertical component.
    print("\nP-WAVE DETECTION")

    for trace in components["Z"]:

        triggers = process_detection(
            trace.data,
            trace.stats.starttime,
            STATION_ID,
            "P",
        )

        print(f"\nTrace: {trace.id}")
        print(f"Samples: {trace.stats.npts}")
        print(f"Sampling rate: {trace.stats.sampling_rate} Hz")
        print(f"Triggers: {len(triggers)}")

        for i, trigger in enumerate(triggers, start=1):

            print(
                f"Trigger {i}: "
                f"time={trigger.detection_time}, "
                f"duration={trigger.duration_sec:.2f}s, "
                f"CFT={trigger.cft_value:.2f}, "
                f"amplitude={trigger.amplitude:.2f}"
            )

    # Step 5: Test S-wave detection using horizontal component.
    print("\nS-WAVE DETECTION")

    for trace in components["H"]:

        triggers = process_detection(
            trace.data,
            trace.stats.starttime,
            STATION_ID,
            "S",
        )

        print(f"\nTrace: {trace.id}")
        print(f"Samples: {trace.stats.npts}")
        print(f"Sampling rate: {trace.stats.sampling_rate} Hz")
        print(f"Triggers: {len(triggers)}")

        for i, trigger in enumerate(triggers, start=1):

            print(
                f"Trigger {i}: "
                f"time={trigger.detection_time}, "
                f"duration={trigger.duration_sec:.2f}s, "
                f"CFT={trigger.cft_value:.2f}, "
                f"amplitude={trigger.amplitude:.2f}"
            )

    print("\nTEST FINISHED")


if __name__ == "__main__":
    main()
