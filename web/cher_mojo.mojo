from os import abort
from math import nan
from python import Python, PythonObject
from python.bindings import PythonModuleBuilder
from gpu.host import HostBuffer
from buffer import NDBuffer
from utils import IndexList
from european_call import Network, Params


@export
fn PyInit_cher_mojo() -> PythonObject:
    try:
        m = PythonModuleBuilder('cher_mojo')

        _ = m.add_type[TestPath]('TestPath')
                .def_method[TestPath.get_inputs]('get_inputs')
                .def_method[TestPath.get_num_steps]('get_num_steps')
                .def_method[TestPath.get_data]('get_data')

        _ = m.add_type[PythonNetwork]('Network')
                .def_py_init[PythonNetwork.py_init]()
                .def_method[PythonNetwork.run]('run')
                .def_method[PythonNetwork.run_test]('run_test')
                .def_method[PythonNetwork.loss]('loss')
                .def_method[PythonNetwork.test_path]('test_path')
                .def_method[PythonNetwork.update_lr]('update_lr')
                .def_method[PythonNetwork.get_params]('get_params')
                .def_method[PythonNetwork.set_slippage]('set_slippage')
                .def_method[PythonNetwork.reset_counters]('reset_counters')

        return m.finalize()

    except e:
        abort(String('error creating cher_mojo', e))


struct TestPath(
        Representable,
        Movable,
        Stringable,
):

    var inputs:    Int
    var num_steps: Int
    var buffer:    HostBuffer[DType.float32]

    fn __init__(
            out self, 
            inputs: Int, 
            num_steps: Int, 
            var buffer: HostBuffer[DType.float32]
    ):
        self.inputs    = inputs
        self.num_steps = num_steps
        self.buffer    = buffer

    fn __str__(self) -> String:
        return 'TestPath __str__'

    fn __moveinit__(out self, deinit existing: Self):
        self = Self(existing.inputs, existing.num_steps, existing.buffer^)

    fn __repr__(self) -> String:
        return self.__str__()

    @staticmethod
    fn get_inputs(py_self: PythonObject) raises -> PythonObject:
        self_ptr = py_self.downcast_value_ptr[Self]()
        return PythonObject(self_ptr[].inputs)

    @staticmethod
    fn get_num_steps(py_self: PythonObject) raises -> PythonObject:
        self_ptr = py_self.downcast_value_ptr[Self]()
        return PythonObject(self_ptr[].num_steps)

    @staticmethod
    fn get_data(py_self: PythonObject) raises -> PythonObject:
        self_ptr = py_self.downcast_value_ptr[Self]()

        inputs = self_ptr[].inputs
        num_steps = self_ptr[].num_steps
        unsafe_pointer = self_ptr[].buffer.unsafe_ptr()
        index_list = IndexList[2](inputs, num_steps)

        nd_buffer = NDBuffer[DType.float32, 2, MutAnyOrigin](unsafe_pointer, index_list)

        builtins = Python.import_module('builtins')

        row_1 = builtins.list()
        row_2 = builtins.list()
        row_3 = builtins.list()

        for i in range(num_steps):
            row_1.append(nd_buffer[0, i])
            row_2.append(nd_buffer[1, i])
            row_3.append(nd_buffer[2, i])

        res = builtins.list()
        res.append(row_1)
        res.append(row_2)
        res.append(row_3)

        return res

struct PythonNetwork(
        Representable, 
        Movable, 
        Stringable,
):

    var network: Network

    @staticmethod
    fn py_init(
            out self: PythonNetwork,
            args: PythonObject, 
            kwargs: PythonObject,
    ) raises:
        self = Self()

    fn __str__(self) -> String:
        try:
            loss = self.network.loss.value()
        except:
            loss = nan[DType.float32]()

        return String.write(
                '== Network ==\n', 
                'Params:\n'
                'inputs\t',       self.network.params.inputs,       '\n',
                'network_size\t', self.network.params.network_size, '\n',
                'num_steps\t',    self.network.params.num_steps,    '\n',
                'num_paths\t',    self.network.params.num_paths,    '\n',
                'lr\t',           self.network.params.lr,           '\n',
                '\n',
                'State',                                            '\n',
                'Loss\t',         loss,                             '\n',
        )

    fn __init__(out self):
        params = Params(
                inputs       = 3,
                network_size = 16,
                num_steps    = 30,
                num_paths    = 1024,
                lr           = 1e-2,
                lr_d1        = 0.95,
                lr_d2        = 10_000,
                beta1        = 0.9,
                beta2        = 0.999,
                eps          = 1e-8,
                weight_decay = 0.01,
                drift        = 0,
                vol          = 0.2,
                strike       = 1.1,
                slippage     = 0.01,
                seed         = 42,
        )
        try:
            self.network = Network(params)
        except e:
            abort(String(e))

    fn __init__(out self, params: Params):
        try:
            self.network = Network(params)
        except e:
            abort(String(e))

    fn __moveinit__(out self, deinit existing: Self):
        self.network = existing.network^

    fn __repr__(self) -> String:
        return self.__str__()

    @staticmethod
    fn run(py_self: PythonObject) raises:
        self_ptr = py_self.downcast_value_ptr[Self]()
        self_ptr[].network.run()

    @staticmethod
    fn run_test(py_self: PythonObject, test_path: PythonObject) raises:
        self_ptr = py_self.downcast_value_ptr[Self]()
        test_path_ptr = test_path.downcast_value_ptr[TestPath]()

        self_ptr[].network.run_test(test_path_ptr[].buffer)

    @staticmethod
    fn test_path(py_self: PythonObject) raises -> PythonObject:
        self_ptr = py_self.downcast_value_ptr[Self]()
        ref network = self_ptr[].network

        var path = TestPath(
                self_ptr[].network.params.inputs,
                self_ptr[].network.params.num_steps,
                self_ptr[].network.generate_test_path(),
        )

        return PythonObject(alloc=path^)


    @staticmethod
    fn loss(py_self: PythonObject) raises -> PythonObject:
        self_ptr = py_self.downcast_value_ptr[Self]()
        return PythonObject(self_ptr[].network.loss_value())

    @staticmethod
    fn update_lr(py_self: PythonObject, lr: PythonObject) raises:
        self_ptr = py_self.downcast_value_ptr[Self]()
        self_ptr[].network.params.lr = Float32(py=lr)

    @staticmethod
    fn set_slippage(py_self: PythonObject, slippage: PythonObject) raises:
        self_ptr = py_self.downcast_value_ptr[Self]()
        self_ptr[].network.params.slippage = Float32(py=slippage)

    @staticmethod
    fn reset_counters(py_self: PythonObject) raises:
        self_ptr = py_self.downcast_value_ptr[Self]()
        self_ptr[].network.reset_counters()

    @staticmethod
    fn get_params(py_self: PythonObject) raises -> PythonObject:
        self_ptr = py_self.downcast_value_ptr[Self]()
        ref params = self_ptr[].network.params

        builtins = Python.import_module('builtins')
        result = builtins.dict()

        result['inputs']       = params.inputs
        result['network_size'] = params.network_size
        result['steps']        = params.num_steps
        result['paths']        = params.num_paths
        result['lr']           = params.lr
        result['lr_d1']        = params.lr_d1
        result['lr_d2']        = params.lr_d2
        result['beta1']        = params.beta1
        result['beta2']        = params.beta2
        result['eps']          = params.eps
        result['weight_decay'] = params.weight_decay
        result['drift']        = params.drift
        result['vol']          = params.vol
        result['strike']       = params.strike
        result['slippage']     = params.slippage
        result['seed']         = Int(params.seed)

        return result

