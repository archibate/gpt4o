import json

def json_dumps(data):
    return json.dumps(data, ensure_ascii=False, separators=(',', ':'))
