;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -O static-memory-maximum-size=0 -O static-memory-guard-size=4294967295 -O dynamic-memory-guard-size=4294967295"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i32 1)

  (func (export "do_store") (param i32 i32)
    local.get 0
    local.get 1
    i32.store offset=0x1000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load offset=0x1000))

;; wasm[0]::function[0]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       mov     w10, w2
;;       ldr     x11, [x0, #0x68]
;;       ldr     x12, [x0, #0x60]
;;       add     x12, x12, w2, uxtw
;;       add     x12, x12, #1, lsl #12
;;       mov     x13, #0
;;       cmp     x10, x11
;;       csel    x11, x13, x12, hi
;;       csdb
;;       str     w3, [x11]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
;;
;; wasm[0]::function[1]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       mov     w10, w2
;;       ldr     x11, [x0, #0x68]
;;       ldr     x12, [x0, #0x60]
;;       add     x12, x12, w2, uxtw
;;       add     x12, x12, #1, lsl #12
;;       mov     x13, #0
;;       cmp     x10, x11
;;       csel    x11, x13, x12, hi
;;       csdb
;;       ldr     w0, [x11]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
