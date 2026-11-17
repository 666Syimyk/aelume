# Повторы ID и неоднозначности (сгенерировано автоматически + ручные заметки ниже)

- TZ section 24 reuses test ID TC-AI-001 for 6 different tests -> derived IDs TC-AI-001.1..TC-AI-001.6 (pending QA approval)
- TZ section 24 reuses test ID TC-APP-001 for 3 different tests -> derived IDs TC-APP-001.1..TC-APP-001.3 (pending QA approval)
- TZ section 24 reuses test ID TC-AUD-001 for 3 different tests -> derived IDs TC-AUD-001.1..TC-AUD-001.3 (pending QA approval)
- TZ section 24 reuses test ID TC-BCK-001 for 3 different tests -> derived IDs TC-BCK-001.1..TC-BCK-001.3 (pending QA approval)
- TZ section 24 reuses test ID TC-DEV-001 for 3 different tests -> derived IDs TC-DEV-001.1..TC-DEV-001.3 (pending QA approval)
- TZ section 24 reuses test ID TC-DSP-001 for 2 different tests -> derived IDs TC-DSP-001.1..TC-DSP-001.2 (pending QA approval)
- TZ section 24 reuses test ID TC-EOL-001 for 2 different tests -> derived IDs TC-EOL-001.1..TC-EOL-001.2 (pending QA approval)
- TZ section 24 reuses test ID TC-IDN-001 for 3 different tests -> derived IDs TC-IDN-001.1..TC-IDN-001.3 (pending QA approval)
- TZ section 24 reuses test ID TC-MEC-001 for 3 different tests -> derived IDs TC-MEC-001.1..TC-MEC-001.3 (pending QA approval)
- TZ section 24 reuses test ID TC-OPT-001 for 6 different tests -> derived IDs TC-OPT-001.1..TC-OPT-001.6 (pending QA approval)
- TZ section 24 reuses test ID TC-OTA-001 for 2 different tests -> derived IDs TC-OTA-001.1..TC-OTA-001.2 (pending QA approval)
- TZ section 24 reuses test ID TC-RTC-001 for 10 different tests -> derived IDs TC-RTC-001.1..TC-RTC-001.10 (pending QA approval)
- TZ section 24 reuses test ID TC-SEC-001 for 5 different tests -> derived IDs TC-SEC-001.1..TC-SEC-001.5 (pending QA approval)
- TZ section 24 reuses test ID TC-UX-001 for 2 different tests -> derived IDs TC-UX-001.1..TC-UX-001.2 (pending QA approval)

## Ручные наблюдения (не исправляют источник; решения — в docs/decisions/OPEN-DECISIONS.md)

- Коды метрик `NFR-01..NFR-22` (PRD гл. 08 и ТЗ гл. 20.1) пересекаются по форме с требованиями ТЗ `NFR-001..NFR-012`.
- PRD `NFR-03` = AV-синхронизация, а в ТЗ 20.1 `NFR-03` = motion-to-photon p95, AV-синхронизация = `NFR-04`: одинаковый код, разный смысл.
- В трассируемость включены только требования ТЗ (364) и FR/NFR PRD; метрики ТЗ 20.1 описаны в docs/spec/METRICS.md.
