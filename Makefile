# Cocotb Makefile for spline_eval.
#
# Quick start:
#   pip install cocotb
#   # install icarus verilog:
#   #   Windows: choco install iverilog     (or download from bleyer.org/icarus)
#   #   macOS:   brew install icarus-verilog
#   #   Linux:   apt install iverilog
#   python spline_coeff_pack.py    # generates coeff_init.mem / width_init.mem
#   make
#
# To view waveforms after a run:
#   gtkwave sim_build/spline_eval_top.fst

SIM          ?= icarus
TOPLEVEL_LANG = verilog

VERILOG_SOURCES = $(PWD)/spline_eval.sv $(PWD)/spline_eval_top.sv

TOPLEVEL = spline_eval_top
MODULE   = test_spline_eval

# Icarus flags
COMPILE_ARGS += -g2012

# FST waveform dump
SIM_ARGS += -fst

include $(shell cocotb-config --makefiles)/Makefile.sim
