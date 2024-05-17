class MissingTrajectoryException(Exception):
    def __init__(self):
        message = "No trajectory was loaded to run the simulation"
        super().__init__(message)

class NoModelException(Exception):
    def __init__(self):
        message = "No GP model was loaded to run the simulation"
        super().__init__(message)

class InvalidParameterExecption(Exception):
    def __init__(self,message):
        super().__init__(message)

class InvalidCommandLineArgumentException(Exception):
    def __init__(self,message):
        super().__init__(message)