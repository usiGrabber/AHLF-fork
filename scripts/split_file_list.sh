SALT=5325

while IFS= read -r f; do
    hash=$(printf '%s' "$f $SALT" | cksum | awk '{print $1}')
    bucket=$((hash % 2))

    if [ "$bucket" -eq 0 ]; then
        echo "$f" >> bucket0.txt
    else
        echo "$f" >> bucket1.txt
    fi
done
