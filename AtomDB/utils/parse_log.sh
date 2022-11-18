#! /bin/bash

# cd log dir
cd $1

# mkdir data_report
rm -rf ./data_report
mkdir ./data_report
TMP_DIR='./data_report'

# Generate csv
delimeter=$(cat *_validate_report.txt | grep rom | grep tensor | awk -F 'tensor' '{print $1}' | wc -c)

# Flops
cat *_validate_report.txt |
    sed -n '/C-Layers/,/Setting validation data/p' |
    sed 's/Setting validation data...//g' | sed -n '/c_id/,/END/p' | sed '1d' | grep -v scratch | grep -v '-' |
    cut -c -$(($delimeter-1)) |
    awk '{print $1 " " $2 " " $5 " " $6}' | uniq | grep -v  '   ' >$TMP_DIR/flops.csv

# Shapes kernel
cat *_validate_report.txt |
    sed -n '/C-Layers/,/Setting validation data/p' |
    sed 's/Setting validation data...//g' | sed -n '/c_id/,/END/p' | sed '1d' | grep -v scratch | grep -v '-' |
    cut -c $delimeter- | cut -d ':' -f 2- | awk '{print $1 " " $2}' | sed 's/(//g' | sed 's/)//g' | sed 's/,/ /g' |
    grep weights | sed 's/_weights//g' | sort | uniq >$TMP_DIR/shapes_kernel.csv

# Shapes output
cat *_validate_report.txt |
    sed -n '/C-Layers/,/Setting validation data/p' |
    sed 's/Setting validation data...//g' | sed -n '/c_id/,/END/p' | sed '1d' | grep -v scratch | grep -v '-' |
    cut -c $delimeter- | cut -d ':' -f 2- | awk '{print $1 " " $2}' | sed 's/(//g' | sed 's/)//g' | sed 's/,/ /g' |
    grep output | sed 's/_output//g' | sort | uniq >$TMP_DIR/shapes_output.csv

# Shapes input
cat *_validate_report.txt |
    sed -n '/C-Layers/,/Setting validation data/p' |
    sed 's/Setting validation data...//g' | sed -n '/c_id/,/END/p' | sed '1d' | grep -v scratch | grep I: |
    awk '{print $2 " " $9}' | grep -E 'conv|pad|dense' | sed 's/(//g' | sed 's/)//g' | sed 's/,/ /g' | sort | uniq >$TMP_DIR/shapes_input.csv

# Latency
cat *_validate_report.txt |
    sed -n '/c_id  m_id  desc/,/Saving validation data/p' |
    sed '1,2d;$d' | sed '$d' | sed '$d' | sed '$d' |
    awk '{print $1 " " $6}' >$TMP_DIR/latency.csv
