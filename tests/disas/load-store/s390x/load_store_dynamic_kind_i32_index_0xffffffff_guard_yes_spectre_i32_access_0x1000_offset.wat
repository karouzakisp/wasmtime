;;! target = "s390x"
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
;;       lg      %r1, 8(%r2)
;;       lg      %r1, 0(%r1)
;;       la      %r1, 0xa0(%r1)
;;       clgrtle %r15, %r1
;;       stmg    %r6, %r15, 0x30(%r15)
;;       lgr     %r1, %r15
;;       aghi    %r15, -0xa0
;;       stg     %r1, 0(%r15)
;;       llgfr   %r3, %r4
;;       lg      %r4, 0x68(%r2)
;;       lgr     %r6, %r3
;;       ag      %r6, 0x60(%r2)
;;       aghik   %r2, %r6, 0x1000
;;       lghi    %r6, 0
;;       clgr    %r3, %r4
;;       locgrh  %r2, %r6
;;       strv    %r5, 0(%r2)
;;       lmg     %r6, %r15, 0xd0(%r15)
;;       br      %r14
;;
;; wasm[0]::function[1]:
;;       lg      %r1, 8(%r2)
;;       lg      %r1, 0(%r1)
;;       la      %r1, 0xa0(%r1)
;;       clgrtle %r15, %r1
;;       stmg    %r14, %r15, 0x70(%r15)
;;       lgr     %r1, %r15
;;       aghi    %r15, -0xa0
;;       stg     %r1, 0(%r15)
;;       llgfr   %r3, %r4
;;       lg      %r4, 0x68(%r2)
;;       lgr     %r5, %r3
;;       ag      %r5, 0x60(%r2)
;;       aghi    %r5, 0x1000
;;       lghi    %r2, 0
;;       clgr    %r3, %r4
;;       locgrh  %r5, %r2
;;       lrv     %r2, 0(%r5)
;;       lmg     %r14, %r15, 0x110(%r15)
;;       br      %r14
