import os
import json
from tornado import web, gen, ioloop, autoreload

class UploadHandler(web.RequestHandler):
    @gen.coroutine
    def get(self):
        files = os.listdir(os.getcwd() + "/files/")
        self.write(json.dumps(files))

    @gen.coroutine
    def post(self):
        file = self.request.files['file'][0]
        fd = open(os.getcwd() + r"/files/" + file.filename,"wb")
        fd.write(file.body)
        fd.close()
        self.write("<script>window.location.reload(window.history.back());</script>")

app = web.Application([
    web.url(r'/', UploadHandler),
    web.url(r'/files/(.*)', web.StaticFileHandler, {'path': 'files/'}),
])

app.listen(8000)
ioloop.IOLoop.instance().start()
