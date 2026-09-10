import numpy as np
import torch

from kermt.util.utils import load_checkpoint, initialize_weights, get_ffn_layer_names, get_loss_func, build_optimizer, build_lr_scheduler
from torch.utils.data import DataLoader
from task.train import train, load_data
from kermt.data import MolCollator
from kermt.util.nn_utils import param_count_trainable, param_count_total

def test_frozen_encoder(finetune_args):


    args = finetune_args
    args.checkpoint_paths = "reference_pretrained_models/grover_base.pt"
    
    # Freeze the encoder parameters
    args.fine_tune_coff = 0.0
    
    # pin GPU to local rank.
    idx = args.gpu
    if args.gpu is not None:
        torch.cuda.set_device(idx)

    features_scaler, scaler, shared_dict, test_data, train_data, val_data = load_data(args, print, None)


    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Load/build model

    model, _ = load_checkpoint(args.checkpoint_paths, current_args=args, logger=None)
    pretrained_model_state_dict = model.state_dict()

    # Initialize weights of (parts of) the model that are not frozen
    initialize_weights(model=model, distinct_init=args.distinct_init, model_idx=0, init_param_names = get_ffn_layer_names(model))
    print(f"get_ffn_layer_names(model): {get_ffn_layer_names(model)}")

    # Get loss and metric functions
    loss_func = get_loss_func(args, model)

    optimizer = build_optimizer(model, args)

    n_trainable_params = param_count_trainable(model)
    n_total_params = param_count_total(model)
    assert n_trainable_params <= n_total_params, "Number of trainable parameters <= total number of parameters"

    if args.cuda:
        model = model.cuda()

    # Learning rate schedulers
    scheduler = build_lr_scheduler(optimizer, args)

    # Bulid data_loader
    shuffle = True
    mol_collator = MolCollator(shared_dict={}, args=args)
    train_data = DataLoader(train_data,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            num_workers=0,
                            collate_fn=mol_collator)

    # Train for 1 epoch
    n_iter, train_loss = train(
        epoch=0,
        model=model,
        data=train_data,
        loss_func=loss_func,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        n_iter=0,
        shared_dict=shared_dict,
        logger=None,
    )

    new_model_state_dict = model.state_dict()

    # Check that KERMT encoder parameters are frozen
    unchanged_param_names = ["kermt.encoders.node_blocks.0.heads.0.mpn_v.W_h.weight", "kermt.encoders.node_blocks.0.heads.1.mpn_q.W_h.weight"]

    for param_name in unchanged_param_names:
        print(new_model_state_dict[param_name].cpu().norm(), pretrained_model_state_dict[param_name].cpu().norm())
        assert torch.allclose(new_model_state_dict[param_name].cpu(), pretrained_model_state_dict[param_name]), f"Parameter {param_name} is has changed"

    # Check that FFN parameters are not frozen
    changed_param_names = ["mol_atom_from_atom_ffn.1.weight",  "mol_atom_from_atom_ffn.2.weight"]
    for param_name in changed_param_names:
        print(new_model_state_dict[param_name].cpu().norm(), pretrained_model_state_dict[param_name].cpu().norm())
        assert not torch.allclose(new_model_state_dict[param_name].cpu(), pretrained_model_state_dict[param_name]), f"Parameter {param_name} is has not changed"