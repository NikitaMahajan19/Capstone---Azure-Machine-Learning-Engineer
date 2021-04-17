import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

data = {
    "data":
    [
        {
            'Column1': "example_value",
            'Column2': "example_value",
            'Column3': "example_value",
            'Column4': "example_value",
            'Column5': "example_value",
            'Column6': "example_value",
            'Column7': "example_value",
            'Column8': "example_value",
        },
    ],
}

body = str.encode(json.dumps(data))

url = 'http://d7bce907-1689-46e1-8c45-61e6f3739f84.southcentralus.azurecontainer.io/score'
api_key = 'lFBN0oWDNhBsTdAUlxrKXbBAX0YlNVG7' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(json.loads(error.read().decode("utf8", 'ignore')))
