;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -O static-memory-maximum-size=0 -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

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
;;       mov     w6, w2
;;       ldr     x7, [x0, #0x68]
;;       cmp     x6, x7
;;       b.hs    #0x28
;;   18: ldr     x8, [x0, #0x60]
;;       strb    w3, [x8, w2, uxtw]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
;;   28: .byte   0x1f, 0xc1, 0x00, 0x00
;;
;; wasm[0]::function[1]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       mov     w6, w2
;;       ldr     x7, [x0, #0x68]
;;       cmp     x6, x7
;;       b.hs    #0x68
;;   58: ldr     x8, [x0, #0x60]
;;       ldrb    w0, [x8, w2, uxtw]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
;;   68: .byte   0x1f, 0xc1, 0x00, 0x00
