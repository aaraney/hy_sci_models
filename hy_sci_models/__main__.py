#!/usr/bin/env python3

# imports useful for those using interactive mode
import matplotlib.pyplot as plt
import numpy as np
import hy_sci_models as hs

# local imports
# from .io import main

# model, training_loss, validation_loss, train_loader, val_loader, test_loader = hs.io.main()
results = hs.io.main()
# y, y_hat = hs.models.nn.test(model, test_loader)

# # Print off summary stats
# print(hs.utils.summary_stats(y, y_hat))

# # A few diagnostic plots
# hs.utils.plot_train_val_loss(training_loss, validation_loss)
# hs.utils.plot_scatter(y, y_hat)

# plt.show()