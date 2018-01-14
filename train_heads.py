import sys
sys.path.append("..")

from training.train_common import prepare, train, validate, validate_batch, save_network_input_output
from training.ds_generators import DataGeneratorClient, DataIterator
from config import COCOSourceConfig, GetConfig
from head_counter_config import HeadCounterConfig, COCOSourceHeadConfig, MPIISourceHeadConfig, PochtaSourceHeadConfig

use_client_gen = False
batch_size = 10

task = sys.argv[1] if len(sys.argv)>1 else "train"
config_name = sys.argv[2] if len(sys.argv)>2 else "HeadCount"
if config_name=='': config_name="HeadCount"
experiment_name = sys.argv[3] if len(sys.argv)>3 else None
if experiment_name=='': experiment_name=None
epoch = int(sys.argv[4]) if len(sys.argv)>4 and sys.argv[4]!='' else None
pochta_val = 'pochta' in sys.argv


config = GetConfig(config_name)


if use_client_gen:
    train_client = DataGeneratorClient(config, port=5555, host="localhost", hwm=160, batch_size=batch_size)
    val_client = DataGeneratorClient(config, port=5556, host="localhost", hwm=160, batch_size=batch_size)

else:

    train_ds = [ COCOSourceHeadConfig("../dataset/coco_train_dataset.h5"),
                 MPIISourceHeadConfig("../dataset/mpii_train_dataset.h5"),
                 PochtaSourceHeadConfig("../dataset/pochta_train_dataset.h5") ]

    val_ds = [ COCOSourceHeadConfig("../dataset/coco_val_dataset.h5"),
               MPIISourceHeadConfig("../dataset/mpii_val_dataset.h5"),
               PochtaSourceHeadConfig("../dataset/pochta_val_dataset.h5") ]

    # val_ds = [ PochtaSourceHeadConfig("../dataset/brainwash_val_dataset.h5")]

    train_client = DataIterator(config, train_ds, shuffle=True, augment=True, batch_size=batch_size)
    val_client = DataIterator(config, val_ds, shuffle=False, augment=False, batch_size=batch_size)

train_samples = train_client.num_samples()
val_samples = val_client.num_samples()

train_di = train_client.gen()
val_di = val_client.gen()


model, iterations_per_epoch, validation_steps, last_epoch, metrics_id, callbacks_list = \
    prepare(config_name=config_name, exp_id=experiment_name, train_samples = train_samples, val_samples = val_samples, batch_size=batch_size, epoch=epoch)

print("Train samples:", train_samples)
print("Val samples:", val_samples)


if task == "train":
    train(model, train_di, val_di, iterations_per_epoch, validation_steps, last_epoch, use_client_gen, callbacks_list)

elif task == "validate":
    validate(model, val_di, validation_steps, use_client_gen, epoch)

elif task == "validate_batch":
    validate_batch(model, val_di, validation_steps, metrics_id, epoch)

elif task == "save_network_input_output":
    save_network_input_output(model, val_di, validation_steps, metrics_id, batch_size, last_epoch)

elif task == "save_network_input":
    save_network_input_output(None, val_di, validation_steps, metrics_id, batch_size)
