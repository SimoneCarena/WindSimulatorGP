class MissingTrajectory(Exception):
    def __init__(self):
        message = "No trajectory was loaded to run the simulation"
        super().__init__(message)