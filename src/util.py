import matplotlib.pyplot as plt

def visualize(ct_slice, mask_slice=None, label='', cmap=plt.cm.bone):
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    axes[0].imshow(ct_slice, cmap=cmap)
    axes[0].set_title('slice ' + label)
    axes[0].axis('off')

    if mask_slice is not None:
        axes[1].imshow(mask_slice, cmap=cmap)
        axes[1].set_title('mask ' + label)
        axes[1].axis('off')

        axes[2].imshow(ct_slice, cmap=cmap)
        axes[2].imshow(mask_slice, alpha=0.5, cmap=cmap)
        axes[2].set_title('slice + mask ' + label)
        axes[2].axis('off')

    return axes
