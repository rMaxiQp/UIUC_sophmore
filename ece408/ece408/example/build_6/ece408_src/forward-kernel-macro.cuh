#ifndef FORWARD_KERNEL_MACRO_CUH_
#define FORWARD_KERNEL_MACRO_CUH_

#define _LOAD_LOCAL_ARRAY_ \
    t0 = T0[0]; \
    t1 = T0[1]; \
    t2 = T0[2]; \
    t3 = T0[3]; \
    t4 = T0[4]; \
    t5 = T0[28]; \
    t6 = T0[29]; \
    t7 = T0[30]; \
    t8 = T0[31]; \
    t9 = T0[32]; \
    t10 = T0[56]; \
    t11 = T0[57]; \
    t12 = T0[58]; \
    t13 = T0[59]; \
    t14 = T0[60]; \
    t15 = T0[84]; \
    t16 = T0[85]; \
    t17 = T0[86]; \
    t18 = T0[87]; \
    t19 = T0[88]; \
    t20 = T0[112]; \
    t21 = T0[113]; \
    t22 = T0[114]; \
    t23 = T0[115]; \
    t24 = T0[116];

#define _SHITTY_CONV_ \
do{ \
        const DType *M_MASK = (DType*) &MASK[mm]; \
        const DType temp = t0  * M_MASK[0] \
                         + t1  * M_MASK[1] \
                         + t2  * M_MASK[2] \
                         + t3  * M_MASK[3] \
                         + t4  * M_MASK[4] \
                         + t5  * M_MASK[5] \
                         + t6  * M_MASK[6] \
                         + t7  * M_MASK[7] \
                         + t8  * M_MASK[8] \
                         + t9  * M_MASK[9] \
                         + t10 * M_MASK[10] \
                         + t11 * M_MASK[11] \
                         + t12 * M_MASK[12] \
                         + t13 * M_MASK[13] \
                         + t14 * M_MASK[14] \
                         + t15 * M_MASK[15] \
                         + t16 * M_MASK[16] \
                         + t17 * M_MASK[17] \
                         + t18 * M_MASK[18] \
                         + t19 * M_MASK[19] \
                         + t20 * M_MASK[20] \
                         + t21 * M_MASK[21] \
                         + t22 * M_MASK[22] \
                         + t23 * M_MASK[23] \
                         + t24 * M_MASK[24]; \
        y_curr[y_idx] = temp; \
        y_idx += OUT_SIZE; \
        mm += KERNEL_SIZE; \
}while(0)

#define _SHITTY_UNROLL_ \
do { \
	_SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
    _SHITTY_CONV_; \
} while(0)

#endif