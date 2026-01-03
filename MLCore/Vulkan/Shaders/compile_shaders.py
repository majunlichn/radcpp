import os
import platform
import shlex
import shutil
import subprocess
import sys
from enum import Enum

compiler = "glslang"
script_root = os.path.dirname(os.path.abspath(__file__))
source_root = os.path.abspath(script_root)
output_root = sys.argv[1]

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

def compile_tensor_op_fill_constant():
    source = os.path.abspath(source_root + "/TensorOp/ForEach.comp")
    output_dir = output_root + "/TensorOp/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for data_type in DataType:
        output = output_dir + f"/FillConstant-{data_type.name}.spv"
        run(f'glslang "{source}" -o "{output}" -V --target-env spirv1.6 -DFillConstant=Operation -DDATA_TYPE_ID={data_type.value} -I"{source_root}"')
        run(f'spirv-opt {output} -o {output} -O')

def compile_tensor_op_add_scalar():
    source = os.path.abspath(source_root + "/TensorOp/ElementWiseUnary.comp")
    output_dir = output_root + "/TensorOp/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for data_type in DataType:
        output = output_dir + f"/AddScalar-{data_type.name}.spv"
        run(f'glslang "{source}" -o "{output}" -V --target-env spirv1.6 -DAddScalar=Operation -DDATA_TYPE_ID={data_type.value} -I"{source_root}"')
        run(f'spirv-opt {output} -o {output} -O')

def compile_tensor_op_subtract_scalar():
    source = os.path.abspath(source_root + "/TensorOp/ElementWiseUnary.comp")
    output_dir = output_root + "/TensorOp/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for data_type in DataType:
        output = output_dir + f"/SubtractScalar-{data_type.name}.spv"
        run(f'glslang "{source}" -o "{output}" -V --target-env spirv1.6 -DSubtractScalar=Operation -DDATA_TYPE_ID={data_type.value} -I"{source_root}"')
        run(f'spirv-opt {output} -o {output} -O')

def compile_tensor_op_add():
    source = os.path.abspath(source_root + "/TensorOp/ElementWiseBinary.comp")
    output_dir = output_root + "/TensorOp/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for data_type in DataType:
        output = output_dir + f"/Add-{data_type.name}.spv"
        run(f'glslang "{source}" -o "{output}" -V --target-env spirv1.6 -DAdd=Operation -DDATA_TYPE_ID={data_type.value} -I"{source_root}"')
        run(f'spirv-opt {output} -o {output} -O')

def compile_tensor_op_subtract():
    source = os.path.abspath(source_root + "/TensorOp/ElementWiseBinary.comp")
    output_dir = output_root + "/TensorOp/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for data_type in DataType:
        output = output_dir + f"/Subtract-{data_type.name}.spv"
        run(f'glslang "{source}" -o "{output}" -V --target-env spirv1.6 -DSubtract=Operation -DDATA_TYPE_ID={data_type.value} -I"{source_root}"')
        run(f'spirv-opt {output} -o {output} -O')

def main() -> int:
    output_dir = sys.argv[1]
    err = 0
    try:
        original_working_dir = os.getcwd()
        compile_tensor_op_fill_constant()
        compile_tensor_op_add_scalar()
        compile_tensor_op_subtract_scalar()
        compile_tensor_op_add()
        compile_tensor_op_subtract()
    except Exception as e:
        print(e)
        err = -1
    finally:
        chdir(original_working_dir)
    return err

if __name__ == '__main__':
    sys.exit(main())
