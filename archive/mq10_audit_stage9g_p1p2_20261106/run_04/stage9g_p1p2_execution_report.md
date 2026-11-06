# Stage9G P1/P2 완료 보고

2026-11-06 KST. **완료: 신규 baseline 4 + L 36 = 40배치·160,000궤적. P3 미실행.** 실행 종료 2026-11-06 01:14:36 KST.

P1 8배치·32,000궤적 뒤 P2 추가 32배치를 연속 실행했다. P1의 L 4배치를 재사용하여 P2 L 36배치를 완결했다. baseline의 나머지 32배치는 Stage7 저장 결과를 읽었으며 재계산하지 않았다.

CFI0–4 모두 PASS. j0/73/199만 index-first로 읽어 기존 params_from_draw 변환·raw sigma_sess·shape·sigma epsilon을 유지했다. 선택 입력/청크 43개와 실제 사용 코드 9개의 기존 manifest 연결을 확인했다. full P 저장 사본 대조를 주장하지 않는다. L은 _lagged만 변경했고 고정배열 및 독립 구동식 검산을 통과했다.

4 baseline sentinel은 저장 Stage7 전체 schema·dtype·shape·float64 비트·무한대·이산 상태가 정확 일치했다. 신규 B/L의 전체 입력·난수 bit 비교도 PASS다. 과거 full 난수 배열은 저장되어 있지 않아 과거 전체 난수의 직접 대조를 주장하지 않는다. strict sampler 허용오차, u 복원 2e-11, 생존 범위/비증가 1e-12를 그대로 유지했다.

|모형|n|L−B 포획률 변화 범위 (pp, 6사례)|B의 12−8 범위 (pp, 3 draw)|L의 12−8 범위 (pp, 3 draw)|
|---|---:|---:|---:|---:|
|original|0.8|-0.325 ~ 0.000|-4.775 ~ -4.125|-4.550 ~ -4.125|
|original|1.0|-0.450 ~ 0.000|-4.750 ~ -3.050|-4.375 ~ -3.000|
|original|1.2|-0.425 ~ 0.000|-4.475 ~ -2.525|-4.125 ~ -2.500|
|calibrated|0.8|-1.025 ~ 0.000|-3.775 ~ -2.750|-3.675 ~ -1.800|
|calibrated|1.0|-1.025 ~ 0.000|-2.550 ~ -1.750|-2.550 ~ -0.850|
|calibrated|1.2|-1.325 ~ 0.000|-2.775 ~ -1.775|-2.675 ~ -0.475|

고정 18개 model×n×draw 사례 모두 L에서도 v12 포획률이 v8보다 낮았다. L−B 포획률 변화는 전체 36사례에서 −1.325~0 pp였다. 이는 선택된 3 draw·고정 MC a의 기술 결과이며 posterior 전체 강건성·같은 개체의 인과효과·생물학적 정답을 뜻하지 않는다. G/H2를 재판정하지 않았다.

M0는 144,000개 L 궤적 모두에서 B와 달랐다. 평균 F0 변화 범위는 −0.762228~−0.164452다. logZ<log(1e−8) 사례는 B 5,655개, L 0개였고 모두 strict 검사를 통과했다. 감지·조건화·검열 상세는 paired_screening_results.csv와 conditioning NPZ에 저장했다.

성공 실행 시작부터 과학 완료까지 37.372초; 40배치 simulation wall 27.925초, CPU 27.531초, peak RSS 227.72 MiB. 구간별 자원은 resource_usage.csv, 합계와 수치 최대 오차는 resource_summary.json. 운영 상한 1GiB·디스크 여유 5GB를 유지했다.

준비 오류는 과학 계산 없이 수정했다: 원본 manifest의 역슬래시 키를 slash로 정규화하고 충돌/없는 키를 거부; platform.py의 shell `ver` 확인을 sys.getwindowsversion 기반 읽기로 대체; UTF-8 명시. run_02는 버전 확인 subprocess 차단, run_03은 cp949 decode 오류로 각각 0배치에서 종료했다. run_04만 과학 40배치를 실행했다. 기존 실패 기록은 보존했다.

현재 환경은 Stage7과 Python/패키지/OS 값이 같고 스레드 환경변수만 1로 명시했다. 차이는 기록하고 baseline 정확 재현으로 검증했다. 전수 감사·WRITE_DAC 검사·권한 변경·외부 원본 쓰기·P3는 수행하지 않았다. 마지막으로 소비 입력 43개·사용 코드 9개·선택 baseline 36개와 sidecar의 보존만 확인했다. Python audit guard는 Python 이벤트 범위의 보호이며 전체 native I/O를 증명하는 전수 검사는 아니다.

재사용: run_04의 검증 완료 코드·40 checkpoint·36 conditioning 결과·CSV·Gate·모수 provenance·자원 기록. 이 실행은 완결되어 재개하거나 다시 계산할 필요가 없다. P3는 금지 범위로 남는다.
