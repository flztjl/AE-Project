import numpy as np
from itertools import product
from sklearn.metrics import confusion_matrix
from sklearn.utils import check_matplotlib_support
from sklearn.utils import deprecated
from sklearn.utils.multiclass import unique_labels
from sklearn.base import is_classifier


class ConfusionMatrixDisplay:
    def __init__(self, confusion_matrix, *, display_labels=None):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    def plot(
            self,
            *,
            include_values=True,
            cmap="viridis",
            xticks_rotation="horizontal",
            values_format=None,
            ax=None,
            colorbar=True,
    ):

        check_matplotlib_support("ConfusionMatrixDisplay.plot")
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        self.text_ = None
        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(1.0)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)

            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0

            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min

                if values_format is None:
                    text_cm = format(cm[i, j], ".2g")
                    if cm.dtype.kind != "f":
                        text_d = format(cm[i, j], "d")
                        if len(text_d) < len(text_cm):
                            text_cm = text_d
                else:
                    text_cm = format(cm[i, j], values_format)

                self.text_[i, j] = ax.text(
                    j, i, text_cm, ha="center", va="center", color=color
                )

        if self.display_labels is None:
            display_labels = np.arange(n_classes)
        else:
            display_labels = self.display_labels
        if colorbar:
            fig.colorbar(self.im_, ax=ax)
        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=display_labels,
            yticklabels=display_labels,
            ylabel="True label",
            xlabel="Predicted label",
        )

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self

    @classmethod
    def from_estimator(
            cls,
            estimator,
            X,
            y,
            *,
            labels=None,
            sample_weight=None,
            normalize=None,
            display_labels=None,
            include_values=True,
            xticks_rotation="horizontal",
            values_format=None,
            cmap="viridis",
            ax=None,
            colorbar=True,
    ):

        method_name = f"{cls.__name__}.from_estimator"
        check_matplotlib_support(method_name)
        if not is_classifier(estimator):
            raise ValueError(f"{method_name} only supports classifiers")
        y_pred = estimator.predict(X)

        return cls.from_predictions(
            y,
            y_pred,
            sample_weight=sample_weight,
            labels=labels,
            normalize=normalize,
            display_labels=display_labels,
            include_values=include_values,
            cmap=cmap,
            ax=ax,
            xticks_rotation=xticks_rotation,
            values_format=values_format,
            colorbar=colorbar,
        )

    @classmethod
    def from_predictions(
            cls,
            y_true,
            y_pred,
            *,
            labels=None,
            sample_weight=None,
            normalize=None,
            display_labels=None,
            include_values=True,
            xticks_rotation="horizontal",
            values_format=None,
            cmap="viridis",
            ax=None,
            colorbar=True,
    ):
        """Plot Confusion Matrix given true and predicted labels.

        Read more in the :ref:`User Guide <confusion_matrix>`.

        .. versionadded:: 0.24

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True labels.

        y_pred : array-like of shape (n_samples,)
            The predicted labels given by the method `predict` of an
            classifier.

        labels : array-like of shape (n_classes,), default=None
            List of labels to index the confusion matrix. This may be used to
            reorder or select a subset of labels. If `None` is given, those
            that appear at least once in `y_true` or `y_pred` are used in
            sorted order.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        normalize : {'true', 'pred', 'all'}, default=None
            Either to normalize the counts display in the matrix:

            - if `'true'`, the confusion matrix is normalized over the true
              conditions (e.g. rows);
            - if `'pred'`, the confusion matrix is normalized over the
              predicted conditions (e.g. columns);
            - if `'all'`, the confusion matrix is normalized by the total
              number of samples;
            - if `None` (default), the confusion matrix will not be normalized.

        display_labels : array-like of shape (n_classes,), default=None
            Target names used for plotting. By default, `labels` will be used
            if it is defined, otherwise the unique labels of `y_true` and
            `y_pred` will be used.

        include_values : bool, default=True
            Includes values in confusion matrix.

        xticks_rotation : {'vertical', 'horizontal'} or float, \
                default='horizontal'
            Rotation of xtick labels.

        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`, the
            format specification is 'd' or '.2g' whichever is shorter.

        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.

        ax : matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        colorbar : bool, default=True
            Whether or not to add a colorbar to the plot.

        Returns
        -------
        display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`

        See Also
        --------
        ConfusionMatrixDisplay.from_estimator : Plot the confusion matrix
            given an estimator, the data, and the label.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.metrics import ConfusionMatrixDisplay
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.svm import SVC
        >>> X, y = make_classification(random_state=0)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...         X, y, random_state=0)
        >>> clf = SVC(random_state=0)
        >>> clf.fit(X_train, y_train)
        SVC(random_state=0)
        >>> y_pred = clf.predict(X_test)
        >>> ConfusionMatrixDisplay.from_predictions(
        ...    y_test, y_pred)
        <...>
        >>> plt.show()
        """
        check_matplotlib_support(f"{cls.__name__}.from_predictions")

        if display_labels is None:
            if labels is None:
                display_labels = unique_labels(y_true, y_pred)
            else:
                display_labels = labels

        cm = confusion_matrix(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            labels=labels,
            normalize=normalize,
        )

        cm_shape = cm.shape[0]
        cm_reversed = cm
        for i in range(0, cm_shape):
            cm_reversed[i, :] = cm[cm_shape - i - 1, :]

        disp = cls(confusion_matrix=cm_reversed, display_labels=display_labels)

        return disp.plot(
            include_values=include_values,
            cmap=cmap,
            ax=ax,
            xticks_rotation=xticks_rotation,
            values_format=values_format,
            colorbar=colorbar,
        )


@deprecated(
    "Function `plot_confusion_matrix` is deprecated in 1.0 and will be "
    "removed in 1.2. Use one of the class methods: "
    "ConfusionMatrixDisplay.from_predictions or "
    "ConfusionMatrixDisplay.from_estimator."
)
def plot_confusion_matrix(
        estimator,
        X,
        y_true,
        *,
        labels=None,
        sample_weight=None,
        normalize=None,
        display_labels=None,
        include_values=True,
        xticks_rotation="horizontal",
        values_format=None,
        cmap="viridis",
        ax=None,
        colorbar=True,
):

    check_matplotlib_support("plot_confusion_matrix")

    if not is_classifier(estimator):
        raise ValueError("plot_confusion_matrix only supports classifiers")

    y_pred = estimator.predict(X)
    cm = confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight, labels=labels, normalize=normalize
    )

    if display_labels is None:
        if labels is None:
            display_labels = unique_labels(y_true, y_pred)
        else:
            display_labels = labels

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    return disp.plot(
        include_values=include_values,
        cmap=cmap,
        ax=ax,
        xticks_rotation=xticks_rotation,
        values_format=values_format,
        colorbar=colorbar,
    )