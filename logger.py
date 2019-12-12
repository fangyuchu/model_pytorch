import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# sys.stdout = Logger('../data/test.log', sys.stdout)
# sys.stderr = Logger('../data/test.log', sys.stderr)     # redirect std err, if necessary