#!/bin/bash
find ~/ml -type d -name "venv" -o -name ".venv" -o -name "env" -o -name "tf_env" | while read -r env; do
    echo "Удаляю виртуальное окружение: $env"
    rm -rf "$env"
done