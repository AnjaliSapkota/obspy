
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from pathlib import Path

client = Client("https://seiscomp.alertnepal.online")

network = "NK"
station = "KKN"
location = "*"
channel = "BHZ"

start_date = "2026-09-01"
end_date = "2026-09-20" 

# Output folder
output_dir = Path("data")
output_dir.mkdir(parents=True, exist_ok=True)
prefix = f"eq_{station.lower()}"
start_date = UTCDateTime(start_date)
end_date = UTCDateTime(end_date)
current = start_date

while current < end_date:

    next_day = UTCDateTime(year=current.year,julday=current.julday) + 86400

    request_end = min(next_day, end_date)

    filename = (f"{prefix}{current.year}"f"{current.month:02d}{current.day:02d}.mseed")

    output_path = output_dir / filename

    print(f"File:      {filename}")
    print(f"Station:   {station}")
    print(f"Channel:   {channel}")

    try:
        st = client.get_waveforms(network=network, station=station, location=location, channel=channel, starttime=current, endtime=request_end,)

        if len(st) == 0:
            print("No data returned.")
        else:
            st.write(str(output_path), format="MSEED")

    except Exception as exc:
        print(f"Download failed: {exc}")

    current = next_day

print("\nDownload completed.")
