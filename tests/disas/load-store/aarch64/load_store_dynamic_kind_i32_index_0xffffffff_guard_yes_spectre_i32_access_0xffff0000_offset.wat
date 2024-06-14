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
    i32.store offset=0xffff0000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load offset=0xffff0000))

;; wasm[0]::function[0]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       mov     w11, w2
;;       ldr     x12, [x0, #0x68]
;;       ldr     x13, [x0, #0x60]
;;       add     x13, x13, w2, uxtw
;;       mov     x14, #0xffff0000
;;       add     x13, x13, x14
;;       mov     x14, #0
;;       cmp     x11, x12
;;       csel    x12, x14, x13, hi
;;       csdb
;;       str     w3, [x12]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
;;
;; wasm[0]::function[1]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       mov     w11, w2
;;       ldr     x12, [x0, #0x68]
;;       ldr     x13, [x0, #0x60]
;;       add     x13, x13, w2, uxtw
;;       mov     x14, #0xffff0000
;;       add     x13, x13, x14
;;       mov     x14, #0
;;       cmp     x11, x12
;;       csel    x12, x14, x13, hi
;;       csdb
;;       ldr     w0, [x12]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
