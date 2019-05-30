from klein import Klein
from klein import run, route

app = Klein()


@app.route("/initAlgRequest", methods=['POST', 'OPTIONS'])
# @requires_auth
# @check_cors
def status(request):
    data_string = request.content.read().decode('utf-8', 'strict')
    print(type(data_string))
    print(data_string)
    request.setResponseCode(200)


app.run('0.0.0.0', 8080)
