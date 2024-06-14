;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -O static-memory-maximum-size=0 -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i32 1)

  (func (export "do_store") (param i32 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load8_u offset=0))

;; wasm[0]::function[0]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       mov     w9, w2
;;       ldr     x10, [x0, #0x68]
;;       ldr     x11, [x0, #0x60]
;;       add     x11, x11, w2, uxtw
;;       mov     x12, #0
;;       cmp     x9, x10
;;       csel    x10, x12, x11, hs
;;       csdb
;;       strb    w3, [x10]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
;;
;; wasm[0]::function[1]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       mov     w9, w2
;;       ldr     x10, [x0, #0x68]
;;       ldr     x11, [x0, #0x60]
;;       add     x11, x11, w2, uxtw
;;       mov     x12, #0
;;       cmp     x9, x10
;;       csel    x10, x12, x11, hs
;;       csdb
;;       ldrb    w0, [x10]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
