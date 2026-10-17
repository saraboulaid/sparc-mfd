# Dataset validation report

## metr-la
- h5: `data\metr-la\metr-la.h5`
- shape: [34272, 207]
- file_size_mb: 54.4
- file_hash: `64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05`
- timestamp_min: 2012-03-01T00:00:00
- timestamp_max: 2012-06-27T23:55:00
- sampling_interval_minutes: 5.0
- adjacency_shape: [207, 207]
- adj_density: 0.040188
- adj_synthetic: **False**
- distances_rows: 295374
- distances_source: **official_dcrnn**
- distances_synthetic: **False**
- sensor_order_alignment_ok: **True**
- sensor_order_margin: 0.1465
- sensor_alignment_ok: **True**
- data_source: **real_dcrnn**
- synthetic: **False**
- ok_for_full: **True**
- notes:
  - Distances use DCRNN sensor IDs; index mapping via h5 axis0 is required

## pems-bay
- h5: `data\pems-bay\pems-bay.h5`
- shape: [52116, 325]
- file_size_mb: 129.63
- file_hash: `65d69fb0a2323dba9867179eb7af47c8b814186bc459ff0a4937d21614153c8f`
- timestamp_min: 2017-01-01T00:00:00
- timestamp_max: 2017-06-30T23:55:00
- sampling_interval_minutes: 5.0
- adjacency_shape: [325, 325]
- adj_density: 0.025505
- adj_synthetic: **False**
- distances_rows: 8357
- distances_source: **official_dcrnn**
- distances_synthetic: **False**
- sensor_order_alignment_ok: **True**
- sensor_order_margin: 0.1833
- sensor_alignment_ok: **True**
- data_source: **real_dcrnn**
- synthetic: **False**
- ok_for_full: **True**
- notes:
  - Distances use DCRNN sensor IDs; index mapping via h5 axis0 is required
