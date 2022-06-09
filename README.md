- Platform supported Windows, Linux, Mac

# ¿Qué es?
El presente proyecto permite realizar multiplicaciones de matrices a través de diferentes herramientas de paralelización como OMP y CUDA, seleccionando el número de hilos y bloques que serán utilizados según sea el caso.

# ¿Cómo instalar?
**1.** Abra su consola y clone el repositorio
```
https://github.com/jahelsantiago/Matrix-Multiplication-CUDA
```

**2.** Cambie el directorio de trabajo actual a la carpeta donde se encuentra el repositorio
```
cd Matrix-Multiplication-CUDA
```

# ¿Cómo usarlo?
**1.** Compile el proyecto con GNU Compiler Collection (GCC) instalado en su sistema
- Opción 1:
```
make
```
- Opción 2:
```
gcc mtrxMultOMP.c -o mmomp -lm -fopenmp
nvcc Matmul.cu -o mmcuda
```

**2.1.** Ejecute el programa con OMP
```
./mmomp <rows> <cols> <threads>
```
**2.2.** Ejecute el programa con CUDA
```
./mmomp <rows> <cols> <blocks> <threads_per_block>
```

**3.** (opcional) Puede correr los tests para verificar que el programa funciona correctamente

Se recomienda hacer uso de Git Bash en Windows para correr los tests

- Opción 1.1: OMP
```
make omp
```

- Opción 1.2: OMP
```
sh testOmp.sh
```

- Opción 2.1: CUDA
```
make cuda
```

- Opción 2.2: CUDA
```
sh testCuda.sh
```
