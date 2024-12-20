# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import simulator_pb2 as simulator__pb2

GRPC_GENERATED_VERSION = '1.68.1'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in simulator_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class SimulatorServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.LoadTask = channel.unary_unary(
                '/simulator.SimulatorService/LoadTask',
                request_serializer=simulator__pb2.LoadTaskRequest.SerializeToString,
                response_deserializer=simulator__pb2.Empty.FromString,
                _registered_method=True)
        self.InitActionPrimitives = channel.unary_unary(
                '/simulator.SimulatorService/InitActionPrimitives',
                request_serializer=simulator__pb2.Empty.SerializeToString,
                response_deserializer=simulator__pb2.Empty.FromString,
                _registered_method=True)
        self.NavigateToObject = channel.unary_unary(
                '/simulator.SimulatorService/NavigateToObject',
                request_serializer=simulator__pb2.NavigateToObjectRequest.SerializeToString,
                response_deserializer=simulator__pb2.Empty.FromString,
                _registered_method=True)
        self.GetSceneName = channel.unary_unary(
                '/simulator.SimulatorService/GetSceneName',
                request_serializer=simulator__pb2.Empty.SerializeToString,
                response_deserializer=simulator__pb2.SceneNameResponse.FromString,
                _registered_method=True)
        self.GetRobotPos = channel.unary_unary(
                '/simulator.SimulatorService/GetRobotPos',
                request_serializer=simulator__pb2.Empty.SerializeToString,
                response_deserializer=simulator__pb2.RobotPosResponse.FromString,
                _registered_method=True)
        self.Step = channel.unary_unary(
                '/simulator.SimulatorService/Step',
                request_serializer=simulator__pb2.Empty.SerializeToString,
                response_deserializer=simulator__pb2.Empty.FromString,
                _registered_method=True)
        self.GetRGBD = channel.unary_unary(
                '/simulator.SimulatorService/GetRGBD',
                request_serializer=simulator__pb2.Empty.SerializeToString,
                response_deserializer=simulator__pb2.ImageResponse.FromString,
                _registered_method=True)
        self.GetRobotJointStates = channel.unary_unary(
                '/simulator.SimulatorService/GetRobotJointStates',
                request_serializer=simulator__pb2.Empty.SerializeToString,
                response_deserializer=simulator__pb2.GetRobotJointStatesResponse.FromString,
                _registered_method=True)
        self.SetRobotJointStates = channel.unary_unary(
                '/simulator.SimulatorService/SetRobotJointStates',
                request_serializer=simulator__pb2.SetRobotJointStatesRequest.SerializeToString,
                response_deserializer=simulator__pb2.Empty.FromString,
                _registered_method=True)
        self.GetRobotEEFPose = channel.unary_unary(
                '/simulator.SimulatorService/GetRobotEEFPose',
                request_serializer=simulator__pb2.Empty.SerializeToString,
                response_deserializer=simulator__pb2.EEFPoseResponse.FromString,
                _registered_method=True)
        self.GetRelativeEEFPose = channel.unary_unary(
                '/simulator.SimulatorService/GetRelativeEEFPose',
                request_serializer=simulator__pb2.Empty.SerializeToString,
                response_deserializer=simulator__pb2.RelativeEEFPoseResponse.FromString,
                _registered_method=True)
        self.GetTaskObjects = channel.unary_unary(
                '/simulator.SimulatorService/GetTaskObjects',
                request_serializer=simulator__pb2.Empty.SerializeToString,
                response_deserializer=simulator__pb2.TaskObjectsResponse.FromString,
                _registered_method=True)
        self.GraspObject = channel.unary_unary(
                '/simulator.SimulatorService/GraspObject',
                request_serializer=simulator__pb2.GraspObjectRequest.SerializeToString,
                response_deserializer=simulator__pb2.Empty.FromString,
                _registered_method=True)
        self.ReachPose = channel.unary_unary(
                '/simulator.SimulatorService/ReachPose',
                request_serializer=simulator__pb2.ReachPoseRequest.SerializeToString,
                response_deserializer=simulator__pb2.Empty.FromString,
                _registered_method=True)
        self.SaveCameraImage = channel.unary_unary(
                '/simulator.SimulatorService/SaveCameraImage',
                request_serializer=simulator__pb2.SaveCameraImageRequest.SerializeToString,
                response_deserializer=simulator__pb2.Empty.FromString,
                _registered_method=True)


class SimulatorServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def LoadTask(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def InitActionPrimitives(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def NavigateToObject(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetSceneName(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetRobotPos(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Step(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetRGBD(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetRobotJointStates(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetRobotJointStates(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetRobotEEFPose(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetRelativeEEFPose(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetTaskObjects(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GraspObject(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ReachPose(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SaveCameraImage(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SimulatorServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'LoadTask': grpc.unary_unary_rpc_method_handler(
                    servicer.LoadTask,
                    request_deserializer=simulator__pb2.LoadTaskRequest.FromString,
                    response_serializer=simulator__pb2.Empty.SerializeToString,
            ),
            'InitActionPrimitives': grpc.unary_unary_rpc_method_handler(
                    servicer.InitActionPrimitives,
                    request_deserializer=simulator__pb2.Empty.FromString,
                    response_serializer=simulator__pb2.Empty.SerializeToString,
            ),
            'NavigateToObject': grpc.unary_unary_rpc_method_handler(
                    servicer.NavigateToObject,
                    request_deserializer=simulator__pb2.NavigateToObjectRequest.FromString,
                    response_serializer=simulator__pb2.Empty.SerializeToString,
            ),
            'GetSceneName': grpc.unary_unary_rpc_method_handler(
                    servicer.GetSceneName,
                    request_deserializer=simulator__pb2.Empty.FromString,
                    response_serializer=simulator__pb2.SceneNameResponse.SerializeToString,
            ),
            'GetRobotPos': grpc.unary_unary_rpc_method_handler(
                    servicer.GetRobotPos,
                    request_deserializer=simulator__pb2.Empty.FromString,
                    response_serializer=simulator__pb2.RobotPosResponse.SerializeToString,
            ),
            'Step': grpc.unary_unary_rpc_method_handler(
                    servicer.Step,
                    request_deserializer=simulator__pb2.Empty.FromString,
                    response_serializer=simulator__pb2.Empty.SerializeToString,
            ),
            'GetRGBD': grpc.unary_unary_rpc_method_handler(
                    servicer.GetRGBD,
                    request_deserializer=simulator__pb2.Empty.FromString,
                    response_serializer=simulator__pb2.ImageResponse.SerializeToString,
            ),
            'GetRobotJointStates': grpc.unary_unary_rpc_method_handler(
                    servicer.GetRobotJointStates,
                    request_deserializer=simulator__pb2.Empty.FromString,
                    response_serializer=simulator__pb2.GetRobotJointStatesResponse.SerializeToString,
            ),
            'SetRobotJointStates': grpc.unary_unary_rpc_method_handler(
                    servicer.SetRobotJointStates,
                    request_deserializer=simulator__pb2.SetRobotJointStatesRequest.FromString,
                    response_serializer=simulator__pb2.Empty.SerializeToString,
            ),
            'GetRobotEEFPose': grpc.unary_unary_rpc_method_handler(
                    servicer.GetRobotEEFPose,
                    request_deserializer=simulator__pb2.Empty.FromString,
                    response_serializer=simulator__pb2.EEFPoseResponse.SerializeToString,
            ),
            'GetRelativeEEFPose': grpc.unary_unary_rpc_method_handler(
                    servicer.GetRelativeEEFPose,
                    request_deserializer=simulator__pb2.Empty.FromString,
                    response_serializer=simulator__pb2.RelativeEEFPoseResponse.SerializeToString,
            ),
            'GetTaskObjects': grpc.unary_unary_rpc_method_handler(
                    servicer.GetTaskObjects,
                    request_deserializer=simulator__pb2.Empty.FromString,
                    response_serializer=simulator__pb2.TaskObjectsResponse.SerializeToString,
            ),
            'GraspObject': grpc.unary_unary_rpc_method_handler(
                    servicer.GraspObject,
                    request_deserializer=simulator__pb2.GraspObjectRequest.FromString,
                    response_serializer=simulator__pb2.Empty.SerializeToString,
            ),
            'ReachPose': grpc.unary_unary_rpc_method_handler(
                    servicer.ReachPose,
                    request_deserializer=simulator__pb2.ReachPoseRequest.FromString,
                    response_serializer=simulator__pb2.Empty.SerializeToString,
            ),
            'SaveCameraImage': grpc.unary_unary_rpc_method_handler(
                    servicer.SaveCameraImage,
                    request_deserializer=simulator__pb2.SaveCameraImageRequest.FromString,
                    response_serializer=simulator__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'simulator.SimulatorService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('simulator.SimulatorService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class SimulatorService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def LoadTask(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/simulator.SimulatorService/LoadTask',
            simulator__pb2.LoadTaskRequest.SerializeToString,
            simulator__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def InitActionPrimitives(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/simulator.SimulatorService/InitActionPrimitives',
            simulator__pb2.Empty.SerializeToString,
            simulator__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def NavigateToObject(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/simulator.SimulatorService/NavigateToObject',
            simulator__pb2.NavigateToObjectRequest.SerializeToString,
            simulator__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GetSceneName(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/simulator.SimulatorService/GetSceneName',
            simulator__pb2.Empty.SerializeToString,
            simulator__pb2.SceneNameResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GetRobotPos(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/simulator.SimulatorService/GetRobotPos',
            simulator__pb2.Empty.SerializeToString,
            simulator__pb2.RobotPosResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Step(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/simulator.SimulatorService/Step',
            simulator__pb2.Empty.SerializeToString,
            simulator__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GetRGBD(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/simulator.SimulatorService/GetRGBD',
            simulator__pb2.Empty.SerializeToString,
            simulator__pb2.ImageResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GetRobotJointStates(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/simulator.SimulatorService/GetRobotJointStates',
            simulator__pb2.Empty.SerializeToString,
            simulator__pb2.GetRobotJointStatesResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def SetRobotJointStates(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/simulator.SimulatorService/SetRobotJointStates',
            simulator__pb2.SetRobotJointStatesRequest.SerializeToString,
            simulator__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GetRobotEEFPose(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/simulator.SimulatorService/GetRobotEEFPose',
            simulator__pb2.Empty.SerializeToString,
            simulator__pb2.EEFPoseResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GetRelativeEEFPose(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/simulator.SimulatorService/GetRelativeEEFPose',
            simulator__pb2.Empty.SerializeToString,
            simulator__pb2.RelativeEEFPoseResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GetTaskObjects(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/simulator.SimulatorService/GetTaskObjects',
            simulator__pb2.Empty.SerializeToString,
            simulator__pb2.TaskObjectsResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GraspObject(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/simulator.SimulatorService/GraspObject',
            simulator__pb2.GraspObjectRequest.SerializeToString,
            simulator__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def ReachPose(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/simulator.SimulatorService/ReachPose',
            simulator__pb2.ReachPoseRequest.SerializeToString,
            simulator__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def SaveCameraImage(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/simulator.SimulatorService/SaveCameraImage',
            simulator__pb2.SaveCameraImageRequest.SerializeToString,
            simulator__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)