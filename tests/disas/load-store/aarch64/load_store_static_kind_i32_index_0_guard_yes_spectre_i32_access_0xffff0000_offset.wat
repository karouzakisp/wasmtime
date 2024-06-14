;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -O static-memory-forced -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

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
;;       ldr     x12, [x0, #0x60]
;;       mov     w11, w2
;;       add     x12, x12, w2, uxtw
;;       mov     x13, #0xffff0000
;;       add     x12, x12, x13
;;       mov     x13, #0
;;       mov     x10, #0xfffc
;;       cmp     x11, x10
;;       csel    x13, x13, x12, hi
;;       csdb
;;       str     w3, [x13]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
;;
;; wasm[0]::function[1]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       ldr     x12, [x0, #0x60]
;;       mov     w11, w2
;;       add     x12, x12, w2, uxtw
;;       mov     x13, #0xffff0000
;;       add     x12, x12, x13
;;       mov     x13, #0
;;       mov     x10, #0xfffc
;;       cmp     x11, x10
;;       csel    x13, x13, x12, hi
;;       csdb
;;       ldr     w0, [x13]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
