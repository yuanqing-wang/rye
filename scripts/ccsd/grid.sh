for coarse_grain_particles in 8; do
for heads in 4; do
for num_rbf in 32 64; do
for hidden_features in 32 64; do
for lr in 1e-2 1e-3; do
for weight_decay in 1e-5 1e-10; do

bsub \
    -q gpuqueue \
    -n 1 \
    -W 4:00 \
    -R "rusage[mem=10] span[ptile=1]" \
    -o %J.stdout \
    python run.py \
    --data ethanol \
    --coarse_grain_particles $coarse_grain_particles \
    --heads $heads \
    --num_rbf $num_rbf \
    --hidden_features $hidden_features \
    --lr $lr \
    --weight_decay $weight_decay

done
done
done
done
done
done