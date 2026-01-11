import os
import platform
import shlex
import shutil
import subprocess
import sys
from enum import Enum
import argparse

compiler = "glslang"
script_root = os.path.dirname(os.path.abspath(__file__))
source_root = os.path.abspath(script_root)
output_root = 'Shaders/'

def comma_separated_list(str):
    if not str:
        return []
    return [token.strip() for token in str.split(',') if token.strip()]

def parse_command_line():
    parser = argparse.ArgumentParser(description='Compile shaders for MLCore Vulkan backend.')
    parser.add_argument('-t', '--targets', type=comma_separated_list,
                    help='compile shaders for specified targets.')
    parser.add_argument('-o', '--output-dir', type=str, default = 'Shaders/',
                    help='specify the output dir.')
    parser.add_argument('-O', '--enable-optimization', action='store_true', default=False,
                    help='optimize the compiled binary with spirv-opt for best performance.')
    return parser.parse_args()

cmd_args = parse_command_line()

class DataType(Enum):
    Float16         = 1
    Float32         = 2
    Float64         = 3
    Sint8           = 4
    Sint16          = 5
    Sint32          = 6
    Sint64          = 7
    Uint8           = 8
    Uint16          = 9
    Uint32          = 10
    Uint64          = 11
    BFloat16        = 12
    Float8E4M3      = 13
    Float8E5M2      = 14


def make_abspath(path:str):
    return os.path.normpath(os.path.abspath(path))

def make_normpath(path:str):
    return os.path.normpath(path)

def chdir(path: str):
    os.chdir(path)
    print("Working dir:", os.getcwd())

def run(command : str, env = os.environ):
    print(command)
    args = shlex.split(command)
    subprocess.run(args, env=env)

def remove_dir(dir : str):
    if os.path.isdir(dir):
        shutil.rmtree(dir)

def compile_tensor_op_fill(dtypes : list[DataType]):
    source = make_abspath(source_root + "/TensorOp/ForEach.comp")
    output_dir = cmd_args.output_dir + "/TensorOp/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for dtype in dtypes:
        output = make_normpath(output_dir + f"/Fill-{dtype.name}.spv")
        run(f'glslang "{source}" -o "{output}" -V --target-env spirv1.6 -DFill=ElementOp -DDATA_TYPE_ID={dtype.value} -I"{source_root}"')
        if cmd_args.enable_optimization:
            run(f'spirv-opt "{output}" -o "{output}" -O')

def compile_tensor_op_element_wise_unary(opname : str, dtypes : list[DataType]):
    source = make_abspath(source_root + "/TensorOp/ElementWiseUnary.comp")
    output_dir = cmd_args.output_dir + "/TensorOp/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for dtype in dtypes:
        output = make_normpath(output_dir + f"/{opname}-{dtype.name}.spv")
        run(f'glslang "{source}" -o "{output}" -V --target-env spirv1.6 -D{opname}=ElementOp -DDATA_TYPE_ID={dtype.value} -I"{source_root}"')
        if cmd_args.enable_optimization:
            run(f'spirv-opt "{output}" -o "{output}" -O')

def compile_tensor_op_element_wise_binary(opname : str, dtypes : list[DataType]):
    source = make_abspath(source_root + "/TensorOp/ElementWiseBinary.comp")
    output_dir = cmd_args.output_dir + "/TensorOp/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for dtype in dtypes:
        output = make_normpath(output_dir + f"/{opname}-{dtype.name}.spv")
        run(f'glslang "{source}" -o "{output}" -V --target-env spirv1.6 -D{opname}=ElementOp -DDATA_TYPE_ID={dtype.value} -I"{source_root}"')
        if cmd_args.enable_optimization:
            run(f'spirv-opt "{output}" -o "{output}" -O')

def main() -> int:
    err = 0

    dtypes = list(DataType)
    computable_dtypes = [
        DataType.Float16, DataType.Float32, DataType.Float64,
        DataType.Sint8, DataType.Sint16, DataType.Sint32, DataType.Sint64,
    ]

    try:
        original_working_dir = os.getcwd()
        compile_all = False
        if cmd_args.targets is None:
            compile_all = True
        if compile_all or any(name in ['Fill'] for name in cmd_args.targets):
            compile_tensor_op_fill(dtypes)
        if compile_all or any(name in ['Add'] for name in cmd_args.targets):
            compile_tensor_op_element_wise_unary('AddScalar', computable_dtypes)
            compile_tensor_op_element_wise_binary('Add', computable_dtypes)
        if compile_all or any(name in ['Subtract'] for name in cmd_args.targets):
            compile_tensor_op_element_wise_unary('SubtractScalar', computable_dtypes)
            compile_tensor_op_element_wise_binary('Subtract', computable_dtypes)
        if compile_all or any(name in ['Multiply'] for name in cmd_args.targets):
            compile_tensor_op_element_wise_unary('MultiplyScalar', computable_dtypes)
            compile_tensor_op_element_wise_binary('Multiply', computable_dtypes)
        if compile_all or any(name in ['Divide'] for name in cmd_args.targets):
            compile_tensor_op_element_wise_unary('DivideScalar', computable_dtypes)
            compile_tensor_op_element_wise_binary('Divide', computable_dtypes)
        if compile_all or any(name in ['Remainder'] for name in cmd_args.targets):
            compile_tensor_op_element_wise_unary('RemainderScalar', computable_dtypes)
            compile_tensor_op_element_wise_binary('Remainder', computable_dtypes)
    except Exception as e:
        print(e)
        err = -1
    finally:
        chdir(original_working_dir)
    return err

if __name__ == '__main__':
    sys.exit(main())
