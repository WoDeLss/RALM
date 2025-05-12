

## Dependencies

- Python>=3.8
- NumPy
- SciPy
- [PyTorch](http://pytorch.org/)>=1.12.~~1~~
- tqdm
- [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)



```shell
# greedy
python eval.py data/hcvrp/hcvrp_v7_100_seed24610.pkl --model outputs/hcvrp_v5_60/epoch-22.pt --obj min-max --decode_strategy greedy --eval_batch_size 1

## Usage

### Generating data

```shell
python generate_data.py 
```

- The `--graph_size`  and `--veh_num`  represent the number of customers , vehicles and generated instances, respectively.

- The  default random seed is 24610, and you can change it in `./generate_data.py`.
- The test set will be stored in `./data/hcvrp/`

### Training

For training HCVRP instances with 40 customers and 3 vehicles (V3-U40):

```shell
python run.py --graph_size 40 --veh_num 3 --baseline rollout --run_name hcvrp_v3_40_rollout --obj min-max
```

```python
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run(get_options())
```

# ~~Evaluation~~

you can test a well-trained model on HCVRP instances with any problem size:

```shell
# greedy 65.55  93
python eval.py data/hcvrp/hcvrp_v3_100_seed26410.pkl --model outputs/hcvrp_v3_40/hcvrp_v3_40_rollout_LJR-Model/RALM.pt --obj min-max --decode_strategy greedy --eval_batch_size 12
```

- The `--model`  represents the directory where the used model is located. 
- The `$filename$.pkl` represents the test set. 
- The `--width` represents sampling number, which is only available when `--decode_strategy` is `sample`.
- The `--eval_batch_size` is set to 1 for serial evaluation.