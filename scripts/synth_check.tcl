set part xczu48dr-ffvg1517-2-e
set top spline_eval

read_verilog -sv [list \
    rtl/spline_eval.sv \
    rtl/spline_eval_top.sv \
]
read_xdc scripts/timing.xdc

synth_design -top $top -part $part
opt_design
place_design
route_design

report_timing_summary -file vivado_reports/timing_summary.rpt
report_timing -nworst 20 -file vivado_reports/timing_worst.rpt
report_utilization -file vivado_reports/utilization.rpt
report_drc -file vivado_reports/drc.rpt
report_clocks -file vivado_reports/clocks.rpt
