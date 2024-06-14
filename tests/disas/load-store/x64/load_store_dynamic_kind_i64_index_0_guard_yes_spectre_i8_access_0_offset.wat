;;! target = "x86_64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -W memory64 -O static-memory-maximum-size=0 -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i64 1)

  (func (export "do_store") (param i64 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load8_u offset=0))

;; wasm[0]::function[0]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    0x68(%rdi), %r10
;;       movq    %rdx, %r8
;;       addq    0x60(%rdi), %r8
;;       xorq    %r11, %r11
;;       cmpq    %r10, %rdx
;;       cmovaeq %r11, %r8
;;       movb    %cl, (%r8)
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;
;; wasm[0]::function[1]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    0x68(%rdi), %r10
;;       movq    %rdx, %r8
;;       addq    0x60(%rdi), %r8
;;       xorq    %r11, %r11
;;       cmpq    %r10, %rdx
;;       cmovaeq %r11, %r8
;;       movzbq  (%r8), %rax
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
