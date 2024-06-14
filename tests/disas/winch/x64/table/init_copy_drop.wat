;;! target = "x86_64"
;;! test = "winch"

(module
  (type (func (result i32)))  ;; type #0
  (import "a" "ef0" (func (result i32)))    ;; index 0
  (import "a" "ef1" (func (result i32)))
  (import "a" "ef2" (func (result i32)))
  (import "a" "ef3" (func (result i32)))
  (import "a" "ef4" (func (result i32)))    ;; index 4
  (table $t0 30 30 funcref)
  (table $t1 30 30 funcref)
  (elem (table $t0) (i32.const 2) func 3 1 4 1)
  (elem funcref
    (ref.func 2) (ref.func 7) (ref.func 1) (ref.func 8))
  (elem (table $t0) (i32.const 12) func 7 5 2 3 6)
  (elem funcref
    (ref.func 5) (ref.func 9) (ref.func 2) (ref.func 7) (ref.func 6))
  (func (result i32) (i32.const 5))  ;; index 5
  (func (result i32) (i32.const 6))
  (func (result i32) (i32.const 7))
  (func (result i32) (i32.const 8))
  (func (result i32) (i32.const 9))  ;; index 9
  (func (export "test")
    (table.init $t0 1 (i32.const 7) (i32.const 0) (i32.const 4))
         (elem.drop 1)
         (table.init $t0 3 (i32.const 15) (i32.const 1) (i32.const 3))
         (elem.drop 3)
         (table.copy $t0 0 (i32.const 20) (i32.const 15) (i32.const 5))
         (table.copy $t0 0 (i32.const 21) (i32.const 29) (i32.const 1))
         (table.copy $t0 0 (i32.const 24) (i32.const 10) (i32.const 1))
         (table.copy $t0 0 (i32.const 13) (i32.const 11) (i32.const 4))
         (table.copy $t0 0 (i32.const 19) (i32.const 20) (i32.const 5)))
  (func (export "check") (param i32) (result i32)
    (call_indirect $t0 (type 0) (local.get 0)))
)
;; wasm[0]::function[5]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    8(%rdi), %r11
;;       movq    (%r11), %r11
;;       addq    $0x10, %r11
;;       cmpq    %rsp, %r11
;;       ja      0x36
;;   1b: movq    %rdi, %r14
;;       subq    $0x10, %rsp
;;       movq    %rdi, 8(%rsp)
;;       movq    %rsi, (%rsp)
;;       movl    $5, %eax
;;       addq    $0x10, %rsp
;;       popq    %rbp
;;       retq
;;   36: ud2
;;
;; wasm[0]::function[6]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    8(%rdi), %r11
;;       movq    (%r11), %r11
;;       addq    $0x10, %r11
;;       cmpq    %rsp, %r11
;;       ja      0x76
;;   5b: movq    %rdi, %r14
;;       subq    $0x10, %rsp
;;       movq    %rdi, 8(%rsp)
;;       movq    %rsi, (%rsp)
;;       movl    $6, %eax
;;       addq    $0x10, %rsp
;;       popq    %rbp
;;       retq
;;   76: ud2
;;
;; wasm[0]::function[7]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    8(%rdi), %r11
;;       movq    (%r11), %r11
;;       addq    $0x10, %r11
;;       cmpq    %rsp, %r11
;;       ja      0xb6
;;   9b: movq    %rdi, %r14
;;       subq    $0x10, %rsp
;;       movq    %rdi, 8(%rsp)
;;       movq    %rsi, (%rsp)
;;       movl    $7, %eax
;;       addq    $0x10, %rsp
;;       popq    %rbp
;;       retq
;;   b6: ud2
;;
;; wasm[0]::function[8]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    8(%rdi), %r11
;;       movq    (%r11), %r11
;;       addq    $0x10, %r11
;;       cmpq    %rsp, %r11
;;       ja      0xf6
;;   db: movq    %rdi, %r14
;;       subq    $0x10, %rsp
;;       movq    %rdi, 8(%rsp)
;;       movq    %rsi, (%rsp)
;;       movl    $8, %eax
;;       addq    $0x10, %rsp
;;       popq    %rbp
;;       retq
;;   f6: ud2
;;
;; wasm[0]::function[9]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    8(%rdi), %r11
;;       movq    (%r11), %r11
;;       addq    $0x10, %r11
;;       cmpq    %rsp, %r11
;;       ja      0x136
;;  11b: movq    %rdi, %r14
;;       subq    $0x10, %rsp
;;       movq    %rdi, 8(%rsp)
;;       movq    %rsi, (%rsp)
;;       movl    $9, %eax
;;       addq    $0x10, %rsp
;;       popq    %rbp
;;       retq
;;  136: ud2
;;
;; wasm[0]::function[10]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    8(%rdi), %r11
;;       movq    (%r11), %r11
;;       addq    $0x10, %r11
;;       cmpq    %rsp, %r11
;;       ja      0x2ad
;;  15b: movq    %rdi, %r14
;;       subq    $0x10, %rsp
;;       movq    %rdi, 8(%rsp)
;;       movq    %rsi, (%rsp)
;;       movq    %r14, %rdi
;;       movl    $0, %esi
;;       movl    $1, %edx
;;       movl    $7, %ecx
;;       movl    $0, %r8d
;;       movl    $4, %r9d
;;       callq   0xaf3
;;       movq    8(%rsp), %r14
;;       movq    %r14, %rdi
;;       movl    $1, %esi
;;       callq   0xb57
;;       movq    8(%rsp), %r14
;;       movq    %r14, %rdi
;;       movl    $0, %esi
;;       movl    $3, %edx
;;       movl    $0xf, %ecx
;;       movl    $1, %r8d
;;       movl    $3, %r9d
;;       callq   0xaf3
;;       movq    8(%rsp), %r14
;;       movq    %r14, %rdi
;;       movl    $3, %esi
;;       callq   0xb57
;;       movq    8(%rsp), %r14
;;       movq    %r14, %rdi
;;       movl    $0, %esi
;;       movl    $0, %edx
;;       movl    $0x14, %ecx
;;       movl    $0xf, %r8d
;;       movl    $5, %r9d
;;       callq   0xb96
;;       movq    8(%rsp), %r14
;;       movq    %r14, %rdi
;;       movl    $0, %esi
;;       movl    $0, %edx
;;       movl    $0x15, %ecx
;;       movl    $0x1d, %r8d
;;       movl    $1, %r9d
;;       callq   0xb96
;;       movq    8(%rsp), %r14
;;       movq    %r14, %rdi
;;       movl    $0, %esi
;;       movl    $0, %edx
;;       movl    $0x18, %ecx
;;       movl    $0xa, %r8d
;;       movl    $1, %r9d
;;       callq   0xb96
;;       movq    8(%rsp), %r14
;;       movq    %r14, %rdi
;;       movl    $0, %esi
;;       movl    $0, %edx
;;       movl    $0xd, %ecx
;;       movl    $0xb, %r8d
;;       movl    $4, %r9d
;;       callq   0xb96
;;       movq    8(%rsp), %r14
;;       movq    %r14, %rdi
;;       movl    $0, %esi
;;       movl    $0, %edx
;;       movl    $0x13, %ecx
;;       movl    $0x14, %r8d
;;       movl    $5, %r9d
;;       callq   0xb96
;;       movq    8(%rsp), %r14
;;       addq    $0x10, %rsp
;;       popq    %rbp
;;       retq
;;  2ad: ud2
;;
;; wasm[0]::function[11]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    8(%rdi), %r11
;;       movq    (%r11), %r11
;;       addq    $0x20, %r11
;;       cmpq    %rsp, %r11
;;       ja      0x39d
;;  2cb: movq    %rdi, %r14
;;       subq    $0x18, %rsp
;;       movq    %rdi, 0x10(%rsp)
;;       movq    %rsi, 8(%rsp)
;;       movl    %edx, 4(%rsp)
;;       movl    4(%rsp), %r11d
;;       subq    $4, %rsp
;;       movl    %r11d, (%rsp)
;;       movl    (%rsp), %ecx
;;       addq    $4, %rsp
;;       movq    %r14, %rdx
;;       movl    0x100(%rdx), %ebx
;;       cmpl    %ebx, %ecx
;;       jae     0x39f
;;  305: movl    %ecx, %r11d
;;       imulq   $8, %r11, %r11
;;       movq    0xf8(%rdx), %rdx
;;       movq    %rdx, %rsi
;;       addq    %r11, %rdx
;;       cmpl    %ebx, %ecx
;;       cmovaeq %rsi, %rdx
;;       movq    (%rdx), %rax
;;       testq   %rax, %rax
;;       jne     0x359
;;  32b: subq    $4, %rsp
;;       movl    %ecx, (%rsp)
;;       subq    $4, %rsp
;;       movq    %r14, %rdi
;;       movl    $0, %esi
;;       movl    4(%rsp), %edx
;;       callq   0xbfa
;;       addq    $4, %rsp
;;       addq    $4, %rsp
;;       movq    0x10(%rsp), %r14
;;       jmp     0x35d
;;  359: andq    $0xfffffffffffffffe, %rax
;;       testq   %rax, %rax
;;       je      0x3a1
;;  366: movq    0x50(%r14), %r11
;;       movl    (%r11), %ecx
;;       movl    0x18(%rax), %edx
;;       cmpl    %edx, %ecx
;;       jne     0x3a3
;;  378: pushq   %rax
;;       popq    %rcx
;;       movq    0x20(%rcx), %rbx
;;       movq    0x10(%rcx), %rdx
;;       subq    $8, %rsp
;;       movq    %rbx, %rdi
;;       movq    %r14, %rsi
;;       callq   *%rdx
;;       addq    $8, %rsp
;;       movq    0x10(%rsp), %r14
;;       addq    $0x18, %rsp
;;       popq    %rbp
;;       retq
;;  39d: ud2
;;  39f: ud2
;;  3a1: ud2
;;  3a3: ud2
