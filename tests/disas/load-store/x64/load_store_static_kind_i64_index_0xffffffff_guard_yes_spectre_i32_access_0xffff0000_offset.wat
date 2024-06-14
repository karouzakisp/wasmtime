;;! target = "x86_64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -W memory64 -O static-memory-forced -O static-memory-guard-size=4294967295 -O dynamic-memory-guard-size=4294967295"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i64 1)

  (func (export "do_store") (param i64 i32)
    local.get 0
    local.get 1
    i32.store offset=0xffff0000)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load offset=0xffff0000))

;; wasm[0]::function[0]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    %rdx, %r11
;;       addq    0x60(%rdi), %r11
;;       movl    $0xffff0000, %esi
;;       leaq    (%r11, %rsi), %r9
;;       xorq    %r11, %r11
;;       cmpq    $0xfffc, %rdx
;;       cmovaq  %r11, %r9
;;       movl    %ecx, (%r9)
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;
;; wasm[0]::function[1]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    %rdx, %r11
;;       addq    0x60(%rdi), %r11
;;       movl    $0xffff0000, %esi
;;       leaq    (%r11, %rsi), %r9
;;       xorq    %r11, %r11
;;       cmpq    $0xfffc, %rdx
;;       cmovaq  %r11, %r9
;;       movl    (%r9), %eax
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
