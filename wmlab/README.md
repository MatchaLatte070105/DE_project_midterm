# wmlab — World Model dựa trên định luật vật lý (bản giảng dạy 0.1)

Bộ công cụ chung cho bốn môn học. Mọi bài tập và đồ án nộp **một file kịch bản** (YAML/JSON) theo cùng một
cấu trúc, được chạy mô phỏng và **kiểm chứng tự động** bằng định luật vật lý.

## Cài đặt
```bash
pip install numpy pyyaml
pip install .            # trong thư mục này; hoặc chạy trực tiếp: python -m wmlab ...
```

## Lệnh
| Lệnh | Việc |
|---|---|
| `python -m wmlab systems` | Liệt kê hệ có sẵn |
| `python -m wmlab validate bai.yaml` | Kiểm tra file đúng cấu trúc |
| `python -m wmlab check bai.yaml -o out/` | Mô phỏng + chạy mọi kiểm tra, ghi `out/<id>_report.json` |
| `python -m wmlab dataset --system oscillator --n 500 -o data.npz` | Sinh bộ dữ liệu ngẫu nhiên cho đồ án mạng nơ-ron |
| `python -m wmlab collect out/ -o registry.csv` | Gom báo cáo, đánh dấu kịch bản đủ điều kiện vào kho |

## Hệ có sẵn
`projectile_drag` (ném xiên có cản) · `oscillator` (dao động tắt dần cưỡng bức, có nghiệm giải tích) ·
`heat1d` (phương trình nhiệt, FTCS/Crank–Nicolson) · `lorenz` (hỗn loạn, số mũ Lyapunov) ·
`langevin` (dao động trong bể nhiệt: phân bố đều năng lượng, hệ thức dao động–đáp ứng) ·
`lorentz` (hạt mang điện trong E, B: bán kính Larmor, trôi E×B; sơ đồ Boris so với RK4) ·
`heat2d` (nhiệt 2D có nguồn, biên cách nhiệt; `sensor_image` mô phỏng ảnh nhiệt đo được).

## Kiểm tra có sẵn
`analytic_match` · `convergence_order` · `energy_conserved` · `energy_nonincreasing` · `max_principle` ·
`stability` · `heat_balance` · `lyapunov_reference` · `equipartition` · `fdt` · `answer_matches`

## Cấu trúc file kịch bản (data contract)
```yaml
id: phy1-proj-range-demo          # duy nhất
author: "Họ tên (MSSV)"
course: PHY1                      # PHY1 | PHY2 | PHY3 | DE | AMCS | RS
system: projectile_drag
law: "định luật áp dụng, viết bằng lời"
params: {...}
initial: {...}
t_span: [0.0, 4.0]
dt: 0.001
question:                         # tuỳ chọn (bắt buộc với bài Physics 1–3)
  text: "..."
  quantity: range                 # đại lượng mô phỏng tính được (xem 'Đại lượng' khi chạy check)
  answer: 38.31                   # đáp án sinh viên tự tính
  unit: m
  tolerance: 0.005                # sai số tương đối
misconceptions: ["..."]           # tuỳ chọn: hiểu lầm phổ biến liên quan
checks: [answer_matches, energy_conserved, {name: analytic_match, tol: 1.0e-6}]
license: CC-BY-4.0                # CC-BY-4.0 | CC0-1.0 | MIT | ALL-RIGHTS-RESERVED
consent_physeval: true            # TỰ NGUYỆN, không ảnh hưởng điểm
```

## Thêm hệ mới (đồ án DE / AMCS)
Viết module trong `wmlab/sims/` với `simulate`, `analytic` (hoặc trả `None`), `quantities`, `sample`
(và `energy` nếu có), rồi đăng ký vào `REGISTRY` trong `wmlab/sims/__init__.py`.

## Ví dụ
`examples/` có 11 kịch bản mẫu (một số được thiết kế để TRƯỢT kiểm tra, ví dụ FTCS mất ổn định) và
`amcs_surrogate_starter.py`: MLP có loss một bước 3e-5 nhưng trôi năng lượng >300% sau 100 s — điểm khởi đầu
cho đồ án "kiểm chứng world model bằng vật lý".

## English messages and student modules (v0.1.1)
```bash
python -m wmlab --lang en check scenario.yaml --module my_system.py     # or: export WMLAB_LANG=en
python -m wmlab --lang en dataset --system my_system --module my_system.py --n 200 -o my_system.npz
```
`--module` loads a student-written system without editing the package. A module may also define
`invariant(p, Y)`, `bounds(p)`, `reference_quantities(p, init)` and `EXTRA_CHECKS`, which enable the checks
`invariant_conserved`, `within_bounds`, `reference_match` and any custom check.
