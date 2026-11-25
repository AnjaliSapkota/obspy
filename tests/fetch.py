from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient


SERVER_URL = "ring.wscada.net:18000"

STATIONS = [
    {"net": "NP", "sta": "EQM10", "cha": "EHZ", "loc": "*", "lat": 28.299517, "lon": 83.960148},
    {"net": "NP", "sta": "EQM26", "cha": "EHZ", "loc": "*", "lat": 27.3936969, "lon": 86.6791885},
    {"net": "NP", "sta": "EQM07", "cha": "EHZ", "loc": "*", "lat": 27.814941, "lon": 86.713401},
    {"net": "NP", "sta": "EQM08", "cha": "EHZ", "loc": "*", "lat": 27.831000, "lon": 86.65},
    {"net": "NP", "sta": "EQM11", "cha": "EHZ", "loc": "*", "lat": 29.2859556, "lon": 81.2741972},
    {"net": "NP", "sta": "EQM22", "cha": "EHZ", "loc": "*", "lat": 28.895805, "lon": 80.227341},
    {"net": "AM", "sta": "R0F23", "cha": "EHZ", "loc": "*", "lat": 28.045045, "lon": 84.072371},
    {"net": "AM", "sta": "R02C2", "cha": "EHZ", "loc": "*", "lat": 27.621622, "lon": 85.320148},
]

client = EasySeedLinkClient(SERVER_URL, autoconnect=False)
client.conn.timeout = 10
client.connect()