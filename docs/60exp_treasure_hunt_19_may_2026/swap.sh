#!/bin/bash
objcopy -I binary -O binary --reverse-bytes=8 $1 $2
