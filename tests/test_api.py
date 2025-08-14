

#schemathesis

#schemathesis run --force-schema-version 30 http://127.0.0.1:5000/openapi/openapi.json
#schemathesis run --force-schema-version 30 https://chatbot.tobiasweise.dev/openapi/openapi.json


#pytest test_api.py


import schemathesis

schema = schemathesis.from_uri("https://chatbot.tobiasweise.dev/openapi/openapi.json", force_schema_version="30")


@schema.parametrize()
def test_api(case):
    case.call_and_validate()







