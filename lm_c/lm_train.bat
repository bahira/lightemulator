@echo off
rem CRITIQUE : les threads libgomp busy-waitent par defaut (spin SSE) et
rem effondrent le debit quand les sections scalaires/AVX alternent.
rem GOMP_SPINCOUNT=0 -> ils dorment. Mesure : x50 sur ce workload.
set GOMP_SPINCOUNT=0
"%~dp0lm_train.exe" %*
