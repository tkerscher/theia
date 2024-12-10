# Creating Custom Components

On this page we will show how to write code for the GPU appropriate for use with
`hephaistos`, the underlying framework responsible for interacting with the GPU.
It introduces its own quirks, that however simplify the process. We start of
with a more general introduction on how to write code, followed by explaining
the aforementioned quirks. The [interfaces](interfaces.md) of each
[component](components.md) type are listed separately.

!!! note
    The way GPU code is written here is specific to `hephaistos` and `theia` as
    they make some assumption about the code in order to provide some
    convenience features.

## GLSL Introduction

Simulation pipeline components contain code to be run on the GPU. Unfortunately,
in our case you cannot use Python for this, but have to use GLSL instead.
Originally designed in 2004 for graphics programming in combination with OpenGL
it has since accumulated some quirks while outdating some older programming
styles. Adding a kind of additional flavour for usage with the Vulkan API
certainly did not help this. While there are some tutorials available on the
internet, they often focus on rendering and/or OpenGl and may be outdated. For
this reason we start with a short introduction to GLSL. Since its style is
similar to C, although missing support for pointers, most parts of it should be
intuitive if you already know another programming language. We instead focus on
its quirks.

!!! info
    We might move in the future to using [slang](https://shader-slang.com/) as
    it features more modern design choices such as OOP and modularization.

### Data Types

GLSL supports the common data types like `int`, `float` and `bool` found in many
other programming languages. Custom composite types are defined by a `struct`:

```GLSL
int a;
float b = 5.0; // (1)!

struct MyType {
    bool flag;
    uint value;
};
MyType t = MyType(true, 42);
```

1. Note that it is not `5.0f`

Additionally, GLSL also provides (column) vector and matrix data types up to four
dimensions that are useful with linear algebra. Vectors are denoted by `vecN`
where `N` is either 2, 3 or 4 indicating the dimension of the vector, e.g.
`vec3`. Similarly, matrices are written as `matNxM`, where `N` is the number
of columns and `M` the number of rows, e.g. `mat3x4`. If these numbers are
equal one can also write simply `matN`. The individual values can be accessed
like arrays, where matrices are treated as 2D with the first index pointing to
the columns. Alternatively for vectors, one can use the `x`, `y`, `z`, `w`
parameters. For more details see the official GLSL wiki on
[data types](https://www.khronos.org/opengl/wiki/Data_Type_(GLSL)).

```GLSL
vec3 v = vec3(1.0, 2.0, 3.0);
//Uses the same value for all parameters
vec4 v2 = vec4(5.0);

float x = v2.x;
//Vector swizzle allows to select a sub-set of given order
vec2 v3 = v2.zy; // = vec2(v2.z, v2.y)

mat4x3 m;
m[1] = vec3(1.0); //sets the second column to all 1.0
m[2][0] = 42.0; //sets first element of third column

//short hand notation for diagonal matrices
mat4 m2 = mat4(5.0); // = 5.0 * identity

//vector matrix multiplication
vec4 v3 = m2 * v2;
vec4 v4 = v2 * m2; // = v2^T * m2 = m2^T * v
```

A specialty of GLSL is the `precise` keyword that prevents the compiler from
rearranging equations. The error caused by the finite precision of floating
point values depend on the exact order of operation. Marking it as precise
makes this error predictable.

```GLSL
precise vec3 v = 3.0 * vec3(1.0) + 5.0;
precise float f = log(exp(2.0));
```

### Functions

Functions are defined the same as in most other programming languages with a
return value and a list of parameters, that are generally copied.[^1]
To make the changes in a parameter visible to the caller, they can be
annotated with `out` or `inout` similar to Fortran. Both copy back the local
variable, but the former creates a new one, while the latter changes a specified
already existing one. Additionally, unlike in C, functions can be overloaded
based on the type and length of the parameter list:

[^1]: The compiler of course may choose to use more reasonable references where
      possible, but is technically not required to do so.

```GLSL
//Typical function definition
int a() {}

//Function can be overloaded on the parameters...
int a(int param) {}
//... but not the return type
float a(float param) {} //ERROR!

//marking parameter as out...
void b(out float result) {
    result = 5.0;
}
//... and how to use it
float f;
b(f); //f = 5.0

//inout parameter behave like references:
void c(inout float f) {
    f *= 2.0;
}
float g = 5.0;
c(g); //g = 10.0
```

The built-in list of standard functions can be found
[here](https://registry.khronos.org/OpenGL-Refpages/gl4/index.php).[^2]
Despite the more common math functions like `sin` and `cos`, this also includes
functions useful for linear algebra like `dot` and `cross`.

[^2]: Functions with the `gl` prefix are part of OpenGL and not GLSL.

### Parameter Structs (UBO)

To make the GPU code configurable, we need a way to pass parameters from the
CPU to the GPU. `hephaistos` and thus `theia` uses
[UBO](https://www.khronos.org/opengl/wiki/Uniform_Buffer_Object) for this, whose
exact details are not important here. They behave in a sense like a global
struct as we will shortly see. Their definition in GLSL are specific to
`hephaistos` and follow this pattern:

```GLSL
//defining a parameter block named "Parameters"
layout(scalar)/*(1)!*/ uniform /*(2)!*/ Parameters /*(3)!*/ {
    //here we define the individual parameters similar to a struct
    vec3 v;
    float f;
    mat3 m;
} params;//(4)!

//usage in code
float func() {
    vec3 v = params.m * v.m;
    return dot(v, v) + params.f;
}
```

1. `layout(scalar)` indicates that the parameters are tightly packed without padding.
   Makes it easier to replicate on the CPU side.
2. `uniform` declares the blocks as UBO
3. `Parameters` is the name of the parameter block and used on the CPU side to
   identify it.
4. `params` is the name of the parameter block on the GPU side. If no name is
   given all parameters of the block spill into the global namespace.

### Input / Output Buffers

UBOs are small and readonly. If you want to process large amount of data you
have to use [SSBO](https://www.khronos.org/opengl/wiki/Shader_Storage_Buffer_Object)
instead. These are rather easy to grasp:

```GLSL
layout(scalar)/*(1)!*/ readonly/*(2)!*/ buffer/*(3)!*/ InputBuffer/*(4)!*/ {
    //content of the buffer is defined similar to a struct
    uint nValues;
    //last element of buffer can be an array without a size specified
    float inValues[];
} inBuffer;//(5)!
layout(scalar) writeonly buffer OutputBuffer {
    float outValues[];
} outBuffer;
```

1. Again, use `layout(scalar)` to prevent GLSL from adding padding.
2. It is good practice to annotate buffers with `readonly` and `writeonly` where
   appropriate as this might allows the compiler to produce better code. If you
   read and write to the buffer omit them.
3. `buffer` marks the following block as buffer.
4. Name of the buffer block used as identification on the CPU side.
5. Name of the buffer on the GPU side. If no name is given all elements of the
   block will be spilled into global namespace.

### Macros

GLSL supports the standard C/C++ macros including `#include`. The
`theia/shader` directory is specified as include directory and you may include
any file within it.

It is good practice to put your shader code in include guards to prevent
compilation errors from multiple includes of the same file:

```GLSL
#ifndef _INCLUDE_MY_CODE
#define _INCLUDE_MY_CODE

//your code goes here

#endif
```

## Defining Components in Python

With the GPU code written, we now need to make it available on the Python side
by creating a new class and deriving from the appropriate component type.
These derive from the base class `SourceCodeMixin` which in turn derives from
`PipelineStage`. The former essentially adds an extra abstract property
`sourceCode` through which the components return their source code.

### Defining Parameters

Each parameter block used in GLSL must be replicated in Python using its
[`ctypes`](https://docs.python.org/3/library/ctypes.html) module. To do so,
you derive from `ctypes.Structure` and define the members of the parameter
block in the `_fields_` list. Definition for GLSL's vector and matrix types
can be found in `hephaistos.glsl`. These are then used in a dictionary mapping
the parameter block names to their definition, which is then passed to the
constructor of the component base class as the `params` named argument:

```GLSL
#ifndef _INCLUDE_MY_CODE
#define _INCLUDE_MY_CODE

layout(scalar) buffer MyParameters {
    vec3 v;
    float f;
    mat4 m;
} myParams;

//[...] your code

#endif
```

```python
from ctypes import Structure, c_float, c_int32
from hephaistos.glsl import vec3, mat4
from theia.light import LightSource

class Parameters(Structure):
    _fields_ = [
        ("myF", c_float),#(1)!
        ("v", vec3),
        ("m", mat4),
    ]

class MyLightSource(LightSource):
    def __init__(self):
        super().__init__(params={"MyParameters": Parameters})
    
    @property
    def sourceCode(self):
        # return the above source code
        return source_code
```

1. You are free to change the name.

!!! warning
    The order of the fields in the parameter block must match in GLSL and Python!

In the background `hephaistos` will allocate and handle all necessary stuff on
CPU and GPU needed for the double buffering expected by a pipeline.[^3] You can
access and alter the parameters either through `getParams` and `setParams`, or
directly by their name:

[^3]: See [pipeline stage](pipeline.md#pipeline-stage)

```python
import numpy as np

myLight = MyLightSource()

myLight.setParams(myF=12.0, v=(1.0, 2.0, 3.0))
myLight.myF += 2.0
print(myLight.myF) # 14.0

myLight.m = np.identity(4)

# print all parameters
print(myLight.getParams())
```

!!! warning
    GLSL defines matrix in column-major memory layout whereas `numpy` uses
    row-major. When assigning a value you therefore have to use the transpose
    instead.

You can mark fields of your parameter block as private by starting their name
with an underscore, i.e. `_privateField`. These will not show up in `getParams`
and will not be accessible as member of the class. You can still set their
values using `setParams`. If instead you want to expose parameters that are only
defined in Python to the pipeline API, you can specify them in the `extras`
parameter of the base class constructor, which expects a set of strings of the
corresponding properties. Finally, before the current values are
used to update the state of the GPU, the `_finishParams` hook is called. This
gives you the chance to set any values that may depend on others or are
expensive to calculate.

```python
class Parameters(Structure):
    _fields_ = [
        ("myF", c_float),
        ("_v", vec3), # marked as private
    ]

class MyLightSource(LightSource):
    def __init__(self):
        super().__init__(
            params={"MyParameters": Parameters},
            extras={"scale"},
        )
    
    @property
    def scale(self):
        return self._scale
    @scale.setter
    def scale(self, value):
        self._scale = value
    
    @property
    def sourceCode(self):
        # return the above source code
        return source_code
    
    def _finishParams(self, config: int): # (1)!
        s = self._scale
        self.setParams(_v=(s, s, s))    
```

1. `config` tells you which config will be updated. See
   [pipeline stage](pipeline.md#pipeline-stage).

### Defining Buffers

Using buffers is more complicated. Since you are allowed to reuse (readonly)
buffers between both configurations, you have to deal with the double buffering
expected by the pipeline yourself. This also includes binding the buffers to the
GPU program and issuing commands to update and/or retrieve data in the buffers
as needed.

The first step is to allocate memory on both the CPU and GPU using `hephaistos`
`*Buffer` and `*Tensor` classes. There are some convenience classes for when
your buffer only consists of a single (unsized) array, such as
`FloatBuffer`/`FloatTensor` and `ArrayBuffer`/`ArrayTensor`. The latter is used
for arrays of a user defined type, i.e. a `struct`, dnd expects a definition
of the type similar to parameters previously shown. In case the data changes,
you usually need to create a separate one for each pipeline configuration to
achieve double buffering.

The GPU does not yet know what to do with these allocations. When your
component gets referenced by most likely a tracer, it will call the `bindParams`
hook on your component passing its shader program to you. You must override this
hook and bind the allocated tensors to the program using `program.bindParams`.
For parameters/UBO `hephaistos` took care of this.

To get data to and from the GPU you have to tell it by issuing the appropriate
`updateTensor` and `retrieveTensor` commands. The pipeline will collect these
during its creation by calling the `run` hook and execute them during each run.
If you only want to update a buffer once, you can use `hehpaistos.execute` to
immediately execute a command.

Finally, we put everything together into an example to illustrate this:

```GLSL
#ifndef _INCLUDE_MY_CODE
#define _INCLUDE_MY_CODE

layout(scalar) uniform Params {
    float scale;
} params;

//you may skip the layout(scalar) if its only scalar data anyway
readonly buffer BufferIn { float value[]; } inBuffer;
writeonly buffer BufferOut { float value[]; } outBuffer;

//we assume this function will be called in parallel for each element
void func(int i) {
    outBuffer.value[i] = params.scale * inBuffer.value[i];
}
```

```python
import numpy as np
import hephaistos as hp

from ctypes import Structure, c_float
from hephaistos import FloatBuffer, FloatTensor, execute

class Params(Structure):
    _fields_ = [("scale", c_float)]

class MyComponent(ComponentType):
    def __init__(self, n):
        super().__init__(params={"Params": Params})
        self.setParams(scale=2.0)

        # allocate memory
        self._inTensor = hp.FloatTensor(n)
        self._outTensor = [hp.FloatTensor(n) for _ in range(2)]
        self._outBuffer = [hp.FloatBuffer(n) for _ in range(2)]

        # fill inBuffer with constant data
        buffer = hp.FloatBuffer(n)
        buffer.numpy()[:] = np.random.rand(n)
        # upload to GPU
        hp.execute(hp.updateTensor(buffer, self._inTensor))
    
    @property
    def sourceCode(self):
        # return the above source code
        return source_code
    
    # function for returning the results (optional)
    def result(self, i):
        return self._outBuffer[i].numpy()
    
    def bindParams(self, program: hp.Program, i: int) -> None:
        # missing this line will break your parameters/UBO!
        super().bindParams(program, i)
        # bind our buffers
        program.bindParams(
            BufferIn=self._inTensor,
            # here happens the double buffering:
            # each config has their own tensor
            BufferOut=self._outTensor[i],
        )
    
    def run(self, i: int) -> list[hp.Command]:
        return [
            # copy the i-th tensor back to CPU into the i-th buffer
            hp.retrieveTensor(self._outBuffer[i], self._outTensor[i]),
            # not needed here, but good practice
            *super().run(i),
        ]
```

## Random Number Generator

While you are unlikely to implement your own random number generator, you will
have to use it in other components. The following functions can be assumed to
be always available:

```GLSL
//normal ones advances i
float random(uint stream, inout uint i);
vec2 random2D(uint stream, inout uint i);
//static ones do not advance i
float random_s(uint stream, uint i);
vec2 random2D_s(uint stream, uint i);
```

All of these functions take the same arguments that are usually passed to the
components and either return a single `float` or a `vec2`. You normally use the
version without `_s` to automatically advance the stream.
