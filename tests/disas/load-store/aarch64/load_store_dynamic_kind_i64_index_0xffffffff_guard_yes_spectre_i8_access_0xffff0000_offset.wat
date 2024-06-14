;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -W memory64 -O static-memory-maximum-size=0 -O static-memory-guard-size=4294967295 -O dynamic-memory-guard-size=4294967295"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i64 1)

  (func (export "do_store") (param i64 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0xffff0000)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load8_u offset=0xffff0000))

;; wasm[0]::function[0]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       ldr     x10, [x0, #0x68]
;;       ldr     x11, [x0, #0x60]
;;       add     x11, x11, x2
;;       mov     x12, #0xffff0000
;;       add     x11, x11, x12
;;       mov     x12, #0
;;       cmp     x2, x10
;;       csel    x11, x12, x11, hi
;;       csdb
;;       strb    w3, [x11]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
;;
;; wasm[0]::function[1]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       ldr     x10, [x0, #0x68]
;;       ldr     x11, [x0, #0x60]
;;       add     x11, x11, x2
;;       mov     x12, #0xffff0000
;;       add     x11, x11, x12
;;       mov     x12, #0
;;       cmp     x2, x10
;;       csel    x11, x12, x11, hi
;;       csdb
;;       ldrb    w0, [x11]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
