#ifndef COMPLEX_H
#define COMPLEX_H

struct Complex32
{
    float16_t real;
    float16_t imag;
};

struct Complex64
{
    float real;
    float imag;
};

struct Complex128
{
    double real;
    double imag;
};

#endif // COMPLEX_H
