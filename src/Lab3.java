import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import javax.imageio.ImageIO;


class Volume {

	// the weight parameters
	public double[] W;

	// the gradients
	public double[] dW;

	// the dimension of the volume dim = [w, h, d]
	public int[] dim = new int[3];

	private static Random rand = new Random();


	/**
	 * Creates a volume and initializes it with random weights
	 * 
	 * @param width
	 *            The width of the volume
	 * @param height
	 *            The height of the volume
	 * @param depth
	 *            The depth of the volume
	 */
	public Volume(int width, int height, int depth) {
		createVolumeWithRandom(width, height, depth);
	}


	/**
	 * Creates a volume with a specified dimension and fills it will a constant
	 * 
	 * @param width
	 *            The width of the volume
	 * @param height
	 *            The height of the volume
	 * @param depth
	 *            The depth of the volume
	 * @param c
	 *            The constant the volume will be filled with
	 */
	public Volume(int width, int height, int depth, double c) {
		createVolumeWithConst(width, height, depth, c);
	}


	/**
	 * Creates a volume and fills it will a constant
	 *
	 * @param v
	 *            The volume whose width, height, and depth will be used for creating the new volume
	 * @param c
	 *            The constant the volume will be filled with
	 */
	public Volume(Volume v, double c) {
		createVolumeWithConst(v.dim[0], v.dim[1], v.dim[2], c);
	}


	/**
	 * Creates a copy of a given volume
	 * 
	 * @param src
	 *            The source volume
	 */
	public Volume(Volume src) {
		for (int i = 0; i < src.dim.length; i++) {
			dim[i] = src.dim[i];
		}

		W = new double[src.W.length];
		for (int i = 0; i < src.W.length; i++) {
			W[i] = src.W[i];
		}

		dW = new double[src.dW.length];
		for (int i = 0; i < src.dW.length; i++) {
			dW[i] = src.dW[i];
		}
	}


	/**
	 * Creates a volume and populates it with a given vector.
	 * 
	 * @param x
	 *            the array of double to populate the volume's W
	 */
	public Volume(double[] x) {
		dim[0] = 1;
		dim[1] = 1;
		dim[2] = x.length;
		W = new double[x.length];
		dW = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			W[i] = x[i];
		}
	}


	/**
	 * Creates a volume give given dimension and populates it with the content of a given array
	 * 
	 * @param x
	 *            the array of double to populate the volume
	 */
	public Volume(int width, int height, int depth, double[] x) {
		dim[0] = width;
		dim[1] = height;
		dim[2] = depth;
		W = new double[dim[0] * dim[1] * dim[2]];
		dW = new double[W.length];
		for (int i = 0; i < W.length; i++) {
			W[i] = x[i];
		}
	}


	private void createVolumeWithRandom(int width, int height, int depth) {
		dim[0] = width;
		dim[1] = height;
		dim[2] = depth;
		W = new double[dim[0] * dim[1] * dim[2]];
		dW = new double[W.length];
		initRandomWeights(W);
	}


	private void createVolumeWithConst(int width, int height, int depth, double c) {
		this.dim[0] = width;
		this.dim[1] = height;
		this.dim[2] = depth;
		W = new double[dim[0] * dim[1] * dim[2]];
		dW = new double[W.length];
		if (c == 0.0) return;
		for (int i = 0; i < W.length; i++) {
			W[i] = c;
		}
	}


	private void initRandomWeights(double[] w) {
		double scale = Math.sqrt(1.0 / ((double) (w.length)));
		for (int i = 0; i < w.length; i++) {
			w[i] = rand.nextGaussian() * scale;
		}
	}


	public int index(int x, int y, int z) {
		return ((dim[0] * y) + x) * dim[2] + z;
	}


	public double get(int x, int y, int z) {
		int i = index(x, y, z);
		return W[i];
	}


	public double getSafe(int x, int y, int z) {
		if (x < 0 || x >= dim[0]) return 0;
		if (y < 0 || y >= dim[1]) return 0;
		if (z < 0 || x >= dim[2]) return 0;
		int i = index(x, y, z);
		return W[i];
	}


	public void set(int x, int y, int z, double v) {
		W[index(x, y, z)] = v;
	}


	public double setSafe(int x, int y, int z, double v) {
		if (x < 0 || x >= dim[0]) return 0;
		if (y < 0 || y >= dim[1]) return 0;
		if (z < 0 || x >= dim[2]) return 0;
		int i = index(x, y, z);
		return W[i] = v;
	}


	public void setAll(double c) {
		for (int i = 0; i < W.length; i++) {
			W[i] = c;
		}
	}


	public int width() {
		return dim[0];
	}


	public int height() {
		return dim[1];
	}


	public int depth() {
		return dim[2];
	}


	public void add(Volume v) {
		add(v.W);
	}


	public void add(double[] d) {
		for (int i = 0; i < W.length; i++) {
			W[i] += d[i];
		}
	}


	public double dot(Volume v) {
		return dot(v.W);
	}


	public double dot(double[] v) {
		double y = 0;
		for (int i = 0; i < W.length; i++) {
			y = y + W[i] * v[i];
		}
		return y;
	}


	public void addScale(Volume v, double scale) {
		addScale(v.W, scale);
	}


	public void addScale(double[] d, double scale) {
		for (int i = 0; i < W.length; i++)
			W[i] += d[i] * scale;

	}


	public void addGrad(int x, int y, int z, double grad) {
		dW[index(x, y, z)] += grad;
	}


	public double getGrad(int x, int y, int z) {
		return dW[index(x, y, z)];
	}


	public void setGrad(int x, int y, int z, double grad) {
		dW[index(x, y, z)] = grad;
	}


	public double dotGrad(Volume v) {
		return dotGrad(v.dW);
	}


	public double dotGrad(double[] v) {
		double y = 0;
		for (int i = 0; i < dW.length; i++) {
			y = y + dW[i] * v[i];
		}
		return y;
	}


	public Volume normalize() {
		double min;
		double max;

		Volume v = new Volume(this);
		double[] u = v.W;

		min = u[0];
		max = u[1];
		for (int i = 0; i < u.length; i++) {
			if (u[i] > max) max = u[i];
			if (u[i] < min) min = u[i];
		}

		double z = (max - min);
		for (int i = 0; i < u.length; i++) {
			u[i] = (u[i] - min) / z;
		}
		return v;
	}


	public Volume normalize(int z) {
		double min;
		double max;
		Volume v = new Volume(this);

		min = v.get(0, 0, z);
		max = min;
		for (int i = 0; i < v.dim[0]; i++) {
			for (int j = 0; j < v.dim[1]; j++) {
				if (v.get(i, j, z) > max) max = v.get(i, j, z);
				if (v.get(i, j, z) < min) min = v.get(i, j, z);
			}
		}

		double m = (max - min);
		for (int i = 0; i < v.dim[0]; i++) {
			for (int j = 0; j < v.dim[1]; j++) {
				double p = v.get(i, j, z);
				p = (p - min) / m;
				v.set(i, j, z, p);
			}
		}
		return v;
	}


	public void zeroMean() {
		for (int z = 0; z < dim[2]; z++) {
			double mean = 0;
			for (int x = 0; x < dim[0]; x++) {
				for (int y = 0; y < dim[1]; y++) {
					mean += W[((dim[0] * y) + x) * dim[2] + z];
				}
			}
			mean /= dim[0] * dim[1];
			for (int x = 0; x < dim[0]; x++) {
				for (int y = 0; y < dim[1]; y++) {
					W[((dim[0] * y) + x) * dim[2] + z] -= mean;
				}
			}
		}
	}

}


class Example {

	public Volume x;
	public Volume y;


	public Example(double[] x, double[] y) {
		this.x = new Volume(x);
		this.y = new Volume(y);
	}


	public Example(Volume x, Volume y) {
		this.x = x;
		this.y = y;
	}

}


class ExampleSet {

	private Example[] _examples;


	public ExampleSet() {
		_examples = new Example[0];
	}


	public ExampleSet(Example[] examples) {
		_examples = new Example[examples.length];
		for (int i = 0; i < examples.length; i++) {
			_examples[i] = examples[i];
		}
	}


	public ExampleSet(List<Example> examples) {
		_examples = new Example[examples.size()];
		for (int i = 0; i < examples.size(); i++) {
			_examples[i] = examples.get(i);
		}
	}


	public void shuffle() {
		for (int i = _examples.length - 1; i >= 1; i--) {
			int j = (int) (Math.random() * i);
			Example t = _examples[i];
			_examples[i] = _examples[j];
			_examples[j] = t;
		}
	}


	public ExampleSet[] split() {
		return null;
	}


	public int count() {
		return _examples.length;
	}


	public Example get(int index) {
		return _examples[index];
	}


	public Example[] examples() {
		return _examples;
	}

}

enum LayerType {
	input, fullconnect, convolution, pool, regression, softmax, leru, sigmoid, tanh, dropout, maxout
}

abstract class Layer {

	public LayerType type;

	public Volume input;

	public Volume output;

	public Volume biases;

	public double bias;

	public int[][] sizes = new int[2][3];

	private Layer _next = null;

	private Layer _last = null;

	public int index;

	private ConvNet _net;


	public Volume forward(double[] x) {
		return null;
	}


	public Volume forward(Volume x) {
		return null;
	}


	public void backward() {

	}


	public double backward(double[] y) {
		return 0;
	}


	public Volume[] response() {
		return new Volume[0];
	}


	public boolean training() {
		return _net.inTraining();
	}


	public void net(ConvNet convNet) {
		_net = convNet;
	}


	public ConvNet net() {
		return _net;
	}


	public Layer next() {
		return _next;
	}


	public void last(Layer l) {
		_last = l;
	}


	public void next(Layer l) {
		_next = l;
	}


	public Layer last() {
		return _last;
	}


	public abstract void connect(Layer l);


	public int inW() {
		return sizes[0][0];
	}


	public int inH() {
		return sizes[0][1];
	}


	public int inD() {
		return sizes[0][2];
	}


	public Layer inW(int w) {
		sizes[0][0] = w;
		return this;
	}


	public Layer inH(int h) {
		sizes[0][1] = h;
		return this;
	}


	public Layer inD(int d) {
		sizes[0][2] = d;
		return this;
	}


	public int outW() {
		return sizes[1][0];
	}


	public int outH() {
		return sizes[1][1];
	}


	public int outD() {
		return sizes[1][2];
	}


	public Layer outW(int w) {
		sizes[1][0] = w;
		return this;
	}


	public Layer outH(int h) {
		sizes[1][1] = h;
		return this;
	}


	public Layer outD(int d) {
		sizes[1][2] = d;
		return this;
	}


	public int inLength() {
		return this.inW() * this.inH() * this.inD();
	}


	public int outLength() {
		return this.outW() * this.outH() * this.outD();
	}

}

abstract class Trainer {

	public interface TrainerEvent {

		boolean call(Trainer trainer);
	}

	protected ConvNet _net;

	protected Example[] _train;

	protected Example[] _tune;

	// the n th example
	protected int _n = 0;

	protected int _epoch = 0;

	protected int _step = 0;

	protected TrainerEvent _onEpoch;

	protected TrainerEvent _onStep;

	protected double _forwardTime;

	protected double _backwardtime;

	protected boolean _stop = false;

	protected double _loss;

	protected double _decayLossL1;

	protected double _decayLossL2;


	public Trainer() {}


	public double forwardTime() {
		return _forwardTime;
	}


	public double backwardTime() {
		return _backwardtime;
	}


	public double costLoss() {
		return _loss;
	}


	public double decayLossL1() {
		return _decayLossL1;
	}


	public double decayLossL2() {
		return _decayLossL2;
	}


	public void net(ConvNet convNet) {
		_net = convNet;
	}


	public ConvNet net() {
		return _net;
	}


	protected Example[] makeTrainExamples(Example[] train) {
		Example[] copy = new Example[train.length];
		for (int i = 0; i < train.length; i++) {
			copy[i] = train[i];
		}
		return copy;
	}


	private void shuffle(Example[] train) {
		for (int i = train.length - 1; i >= 1; i--) {
			int j = (int) (Math.random() * i);
			Example t = train[i];
			train[i] = train[j];
			train[j] = t;
		}
	}


	private Example drawOneExample() {
		Example ex = _train[_n];
		_n++;
		if (_n == _train.length) {
			this.shuffle(_train);
			this.incEpoch();
			_n = 0;
		}
		return ex;
	}


	public int step() {
		return _step;
	}


	public void incStep() {
		_step++;
		if (_onStep != null) {
			_stop = !_onStep.call(this);
		}
	}


	public void incEpoch() {
		_epoch++;
		if (_onEpoch != null) {
			_stop = !_onEpoch.call(this);
			if (_stop) {
				_stop = true;
			}
		}
	}


	public void onEpoch(TrainerEvent callback) {
		_onEpoch = callback;
	}


	public void onStep(TrainerEvent callback) {
		_onStep = callback;
	}


	public int epoch() {
		return _epoch;
	}


	public void train(ConvNet net, Example[] train) {
		this.train(net, train, null);
	}


	public void train(ConvNet net, Example[] train, Example[] tune) {
		_net = net;
		_train = makeTrainExamples(train);
		_tune = tune;
		_stop = false;

		while (_epoch < _net.epochs) {
			Example ex = drawOneExample();
			if (_stop) break;

			_net.inTraining(true);
			this.trainOneExample(net, ex.x.W, ex.y.W);
			_net.inTraining(false);

			this.incStep();
			if (_stop) break;
		}
	}


	protected abstract void trainOneExample(ConvNet net, double[] x, double[] y);

}

class ConvNet {

	private List<Layer> _layerList = new ArrayList<Layer>();

	private boolean _training;

	private Trainer _trainer;

	private Layer[] _layers;

	private Layer _current;

	public int epochs;


	public ConvNet() {}


	public boolean inTraining() {
		return _training;
	}


	public ConvNet addLayer(Layer layer) {
		Layer last = null;
		int index = 0;
		if (_layerList.size() > 0) {
			last = _layerList.get(_layerList.size() - 1);
			index = _layerList.size();
		}

		layer.net(this);
		_layerList.add(layer);
		layer.index = index;

		if (last != null) {
			last.next(layer);
			layer.last(last);
			layer.connect(last);
		}

		// since we modified the layer list we must clear the layer array so it will generate a new one
		_layers = null;
		return this;
	}


	public Layer[] layers() {
		if (_layers == null) {
			_layers = _layerList.toArray(new Layer[_layerList.size()]);
		}
		return _layers;
	}


	public double[] predict(double[] x) {
		_training = false;
		double[] yhat = forward(x);
		return yhat;
	}


	public double[] accuracy(Example[] test) {
		return null;
	}


	public double[] forward(double[] x) {
		Layer[] layers = this.layers();

		Volume act = layers[0].forward(x);
		for (int i = 1; i < layers.length; i++) {
			act = layers[i].forward(act);
		}

		return act.W;
	}


	public double backward(double[] y) {
		Layer[] layers = this.layers();

		double loss = layers[layers.length - 1].backward(y);
		for (int i = layers.length - 2; i >= 0; i--) {
			layers[i].backward();
		}

		return loss;
	}


	public Volume[] response() {
		Layer[] layers = this.layers();
		List<Volume> ret = new ArrayList<Volume>();

		for (int i = 0; i < layers.length; i++) {
			Volume[] r = layers[i].response();
			for (int j = 0; j < r.length; j++) {
				ret.add(r[j]);
			}
		}

		return ret.toArray(new Volume[ret.size()]);
	}


	public void inTraining(boolean training) {
		this._training = training;

	}

}

class ImageUtil {

	public enum LoadOption {
		RGB, RGB_EDGES, GRAY, RGB_GRAY, EDGES,
	}


	public static Volume imageToVolume(BufferedImage image) {
		return imageToVolume(image, LoadOption.RGB);
	}


	public static Volume imageToVolume(BufferedImage image, LoadOption option) {
		int width = image.getWidth();
		int height = image.getHeight();
		Volume v = null;

		switch (option) {

			case RGB:
				// RGB volume
				v = new Volume(width, height, 3);
				for (int i = 0; i < width; i++) {
					for (int j = 0; j < height; j++) {
						Color c = new Color(image.getRGB(i, j));
						v.set(i, j, 0, (((double) c.getRed())) / 255);
						v.set(i, j, 1, (((double) c.getGreen())) / 255);
						v.set(i, j, 2, (((double) c.getBlue())) / 255);
					}
				}
				break;

			case GRAY:
				// Gray scale volume
				v = new Volume(width, height, 1);
				for (int i = 0; i < width; i++) {
					for (int j = 0; j < height; j++) {
						Color c = new Color(image.getRGB(i, j));
						double g = rgbToGrayScale(c.getRed(), c.getGreen(), c.getBlue());
						v.set(i, j, 0, g);
					}
				}
				break;

			case RGB_GRAY:
				// RGB and gray scale volume
				v = new Volume(width, height, 4);
				for (int i = 0; i < width; i++) {
					for (int j = 0; j < height; j++) {
						Color c = new Color(image.getRGB(i, j));
						v.set(i, j, 0, (((double) c.getRed())) / 255);
						v.set(i, j, 1, (((double) c.getGreen())) / 255);
						v.set(i, j, 2, (((double) c.getBlue())) / 255);
						double g = rgbToGrayScale(c.getRed(), c.getGreen(), c.getBlue());
						v.set(i, j, 3, g);
					}
				}
				break;

			case RGB_EDGES:
				// RGB and edges volume
				v = new Volume(width, height, 4);
				for (int i = 0; i < width; i++) {
					for (int j = 0; j < height; j++) {
						Color c = new Color(image.getRGB(i, j));
						v.set(i, j, 0, (((double) c.getRed())) / 255);
						v.set(i, j, 1, (((double) c.getGreen())) / 255);
						v.set(i, j, 2, (((double) c.getBlue())) / 255);
						double g = rgbToGrayScale(c.getRed(), c.getGreen(), c.getBlue());
						v.set(i, j, 3, g);
					}
				}
				v = sobelFilter(v, 3);
				break;

			case EDGES:
				// Gray scale volume
				v = new Volume(width, height, 1);
				for (int i = 0; i < width; i++) {
					for (int j = 0; j < height; j++) {
						Color c = new Color(image.getRGB(i, j));
						double g = rgbToGrayScale(c.getRed(), c.getGreen(), c.getBlue());
						v.set(i, j, 0, g);
					}
				}
				v = sobelFilter(v, 0);

				break;
		}
		return v;
	}


	private static double pixelAt(Volume v, int x, int y, int z) {
		if (x < 0 || x >= v.width()) return 0;
		if (y < 0 || y >= v.width()) return 0;
		if (y < 0 || y >= v.width()) return 0;
		if (z < 0 || z >= v.depth()) return 0;
		return v.get(x, y, z);
	}


	private static Volume sobelFilter(Volume v, int z) {
		Volume u = new Volume(v);
		int[][] sobelX = {
							{ -1, 0, 1 },
							{ -2, 0, 2 },
							{ -1, 0, 1 }
		};

		int[][] sobelY = {
							{ -1, -2, -1 },
							{ 0, 0, 0 },
							{ 1, 2, 1 }
		};

		for (int x = 0; x < v.width(); x++) {
			for (int y = 0; y < v.height(); y++) {
				double px = (sobelX[0][0] * pixelAt(v, x - 1, y - 1, z)) + (sobelX[0][1] * pixelAt(v, x, y - 1, z)) + (sobelX[0][2] * pixelAt(v, x + 1, y - 1, z))
						+ (sobelX[1][0] * pixelAt(v, x - 1, y, z)) + (sobelX[1][1] * pixelAt(v, x, y, z)) + (sobelX[1][2] * pixelAt(v, x + 1, y, z))
						+ (sobelX[2][0] * pixelAt(v, x - 1, y + 1, z)) + (sobelX[2][1] * pixelAt(v, x, y + 1, z)) + (sobelX[2][2] * pixelAt(v, x + 1, y + 1, z));

				double py = (sobelY[0][0] * pixelAt(v, x - 1, y - 1, z)) + (sobelY[0][1] * pixelAt(v, x, y - 1, z)) + (sobelY[0][2] * pixelAt(v, x + 1, y - 1, z))
						+ (sobelY[1][0] * pixelAt(v, x - 1, y, z)) + (sobelY[1][1] * pixelAt(v, x, y, z)) + (sobelY[1][2] * pixelAt(v, x + 1, y, z))
						+ (sobelY[2][0] * pixelAt(v, x - 1, y + 1, z)) + (sobelY[2][1] * pixelAt(v, x, y + 1, z)) + (sobelY[2][2] * pixelAt(v, x + 1, y + 1, z));

				double p = Math.sqrt(px * px + py * py);
				u.set(x, y, z, p);
			}
		}
		u = u.normalize(z);
		return u;
	}


	public static double rgbToGrayScale(int r, int g, int b) {
		double r0 = (double) r / (255);
		double g0 = (double) g / (255);
		double b0 = (double) b / (255);
		double y = .2126 * r0 + .7152 * g0 + .0722 * b0;
		return y;
	}


	public static int rgbToInt(double r, double g, double b) {
		int rgb = (int) (r * 255);
		rgb = (rgb << 8) + (int) (g * 255);
		rgb = (rgb << 8) + (int) (b * 255);
		return rgb;
	}


	private static BufferedImage volumeToImageLayer(Volume v, int layer) {
		BufferedImage image;
		Volume u = v.normalize(layer);
		image = new BufferedImage(u.width(), u.height(), BufferedImage.TYPE_INT_RGB);
		for (int i = 0; i < u.height(); i++) {
			for (int j = 0; j < u.width(); j++) {
				int p = (int) (u.get(j, i, layer) * 255);
				p = p + (p << 8) + (p << 16);
				image.setRGB(j, i, p);
			}
		}
		return image;
	}


	public static BufferedImage volumeToImage(Volume v) {
		BufferedImage image;
		Volume u = v.normalize();

		if (u.depth() > 1) {
			image = new BufferedImage(u.width(), u.height(), BufferedImage.TYPE_INT_RGB);
			for (int i = 0; i < u.height(); i++) {
				for (int j = 0; j < u.width(); j++) {
					double r = u.get(j, i, 0);
					double g = u.get(j, i, 1);
					double b = u.get(j, i, 2);
					image.setRGB(j, i, rgbToInt(r, g, b));
				}
			}
		}
		else {
			image = new BufferedImage(u.width(), u.height(), BufferedImage.TYPE_INT_RGB);
			for (int i = 0; i < u.height(); i++) {
				for (int j = 0; j < u.width(); j++) {
					int p = (int) (u.get(j, i, 0) * 255);
					p = p + (p << 8) + (p << 16);
					image.setRGB(j, i, p);
				}
			}
		}
		return image;
	}


	public static Volume distortImage(Volume image) {
		BufferedImage img = volumeToImage(image);
		img = distortImage(img);
		return imageToVolume(img);
	}


	public static BufferedImage distortImage(BufferedImage image) {

		int r = (int) (Math.random() * 2);
		switch (r) {
			case 0:
				// Flip the image vertically
				AffineTransform tx = AffineTransform.getScaleInstance(1, -1);
				tx.translate(0, -image.getHeight(null));
				AffineTransformOp op = new AffineTransformOp(tx, AffineTransformOp.TYPE_NEAREST_NEIGHBOR);
				image = op.filter(image, null);
				break;

			case 1:
				// Flip the image horizontally  
				tx = AffineTransform.getScaleInstance(-1, 1);
				tx.translate(-image.getWidth(null), 0);
				op = new AffineTransformOp(tx, AffineTransformOp.TYPE_NEAREST_NEIGHBOR);
				image = op.filter(image, null);
				break;
			case 2:
				// Flip the image vertically and horizontally; equivalent to rotating the image 180 degrees
				tx = AffineTransform.getScaleInstance(-1, -1);
				tx.translate(-image.getWidth(null), -image.getHeight(null));
				op = new AffineTransformOp(tx, AffineTransformOp.TYPE_NEAREST_NEIGHBOR);
				image = op.filter(image, null);
				break;
		}

		return image;
	}


	public static void saveImage(Volume v, String fileName) {
		BufferedImage image = volumeToImage(v);
		saveImage(image, fileName);
	}


	public static void saveImage(BufferedImage image, String fileName) {
		File outputfile = new File(fileName);
		try {
			ImageIO.write(image, "png", outputfile);
		}
		catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}


	public static void saveImageLayer(Volume v, int layer, String fileName) {
		BufferedImage image = volumeToImageLayer(v, layer);
		saveImage(image, fileName);
	}


	public static void saveFilters(Volume[] filters, int cols, String fileName) {
		int pad = 1;
		int length = filters.length - 1;
		int rows = length / cols;
		int imageW = filters[0].width();
		int imageH = filters[1].height();

		int extra = length % cols;
		if (extra > 0) rows = rows + 1;
		int width = cols * imageW + pad * (cols + 1);
		int height = rows * imageH + pad * (rows + 1);

		int ox = pad;
		int oy = pad;

		BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

		int c = 0xb3b3ff;
		// c = c + (c << 8) + (c << 16);
		for (int i = 0; i < image.getWidth(); i++) {
			for (int j = 0; j < image.getHeight(); j++) {
				image.setRGB(i, j, c);
			}
		}

		for (int d = 0; d < length; d++) {
			Volume u = filters[d].normalize();
			for (int i = 0; i < u.width(); i++) {
				for (int j = 0; j < u.height(); j++) {
					int r = (int) (u.get(i, j, 0) * 255);
					int g = (int) (u.get(i, j, 1) * 255);
					int b = (int) (u.get(i, j, 2) * 255);
					try {
						image.setRGB(ox + i, oy + j, rgbToInt(r, g, b));
					}
					catch (Exception ex) {
						System.out.print(ex.getMessage());
					}
				}
			}
			if ((d + 1) % cols == 0) {
				oy = oy + imageH + pad;
				ox = pad;
			}
			else {
				ox = ox + imageW + pad;
			}

		}

		image = scaleImage(image, 512, 512);
		saveImage(image, fileName);
	}


	public static void saveVolumeLayers(Volume v, int cols, String fileName) {
		int pad = 1;
		int rows = v.depth() / cols;
		int r = v.depth() % cols;
		if (r > 0) rows = rows + 1;
		int width = cols * v.width() + pad * (cols + 1);
		int height = rows * v.height() + pad * (rows + 1);
		int ox = pad;
		int oy = pad;

		BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

		int c = 0xb3b3ff;
		// c = c + (c << 8) + (c << 16);
		for (int i = 0; i < image.getWidth(); i++) {
			for (int j = 0; j < image.getHeight(); j++) {
				image.setRGB(i, j, c);
			}
		}

		for (int l = 0; l < v.depth(); l++) {
			Volume u = v.normalize(l);
			for (int i = 0; i < u.width(); i++) {
				for (int j = 0; j < u.height(); j++) {
					int p = (int) (u.get(i, j, l) * 255);
					p = p + (p << 8) + (p << 16);
					try {
						image.setRGB(ox + i, oy + j, p);
					}
					catch (Exception ex) {
						System.out.println(ex.getMessage());
					}
				}
			}
			if ((l + 1) % cols == 0) {
				oy = oy + v.height() + pad;
				ox = pad;
			}
			else {
				ox = ox + v.width() + pad;
			}

		}

		image = scaleImage(image, 512, 512);
		saveImage(image, fileName);
	}


	public static Volume loadImage(String fileName, LoadOption options) {
		Volume v = null;
		File inputFile = new File(fileName);
		try {
			BufferedImage image = ImageIO.read(inputFile);
			v = imageToVolume(image, options);
		}
		catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
		return v;
	}


	public static BufferedImage imageToBufferedImage(Image img) {
		if (img instanceof BufferedImage) {
			return (BufferedImage) img;
		}

		BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_RGB);
		Graphics2D g = bimage.createGraphics();
		g.drawImage(img, 0, 0, null);
		g.dispose();

		return bimage;
	}


	public static BufferedImage scaleImage(BufferedImage img, int width, int height) {
		Image scaledImage = img.getScaledInstance(width, height, java.awt.Image.SCALE_DEFAULT);
		return imageToBufferedImage(scaledImage);
	}

}

class ImageDataSetReader {

	private String _fileDir;
	private int _size;
	private String[] _cats;
	private ImageUtil.LoadOption _option;


	public ImageDataSetReader(String fileDir, String[] categories, int size) {
		init(fileDir, categories, size, ImageUtil.LoadOption.RGB);

	}


	public ImageDataSetReader(String fileDir, String[] categories, int size, ImageUtil.LoadOption option) {
		init(fileDir, categories, size, option);
	}


	private void init(String fileDir, String[] categories, int size, ImageUtil.LoadOption option) {
		_fileDir = fileDir;
		_size = size;
		_cats = categories;
		_option = option;
		for (int i = 0; i < categories.length; i++) {
			categories[i] = categories[i]	.trim()
											.toLowerCase();
		}
	}


	private int getCatNumber(String name) {
		name = name	.trim()
					.toLowerCase();
		for (int i = 0; i < _cats.length; i++) {
			if (_cats[i].compareTo(name) == 0) return i;
		}
		return -1;
	}


	private Volume imageNameToVolume(String name) {

		name = name.toLowerCase();
		int cat = 0;

		for (int i = 0; i < _cats.length; i++) {
			if (name.contains(_cats[i])) {
				cat = i;
				break;
			}
		}

		double[] y = new double[_cats.length];
		y[cat] = 1.0;
		return new Volume(y);
	}


	private Example imageToExample(String name, BufferedImage image, ImageUtil.LoadOption options) {
		Volume x = ImageUtil.imageToVolume(image, options);
		x.zeroMean();
		Volume y = imageNameToVolume(name);
		return new Example(x, y);
	}


	public ExampleSet readDataSet() {

		List<Example> examples = new ArrayList<Example>();

		File dir = new File(_fileDir);

		if (_size <= 0) _size = 32;

		for (File file : dir.listFiles()) {

			if (!file.isFile()) continue;

			String fileName = file	.getName()
									.toLowerCase();
			if (!(fileName.endsWith(".jpg") || fileName.endsWith(".jpeg") || fileName.endsWith(".png"))) continue;

			try {
				BufferedImage img = ImageIO.read(file);
				if (img.getWidth() != _size || img.getHeight() != _size) {
					img = ImageUtil.scaleImage(img, _size, _size);
				}
				Example e = imageToExample(fileName, img, _option);
				examples.add(e);
				// saveImageLayer(e.x, 3, "./bin/images/z" + fileName + ".png");
				// ImageUtil.(e.x, "./bin/images/z" + fileName + "_e.png");
			}
			catch (IOException ex) {
				System.err.println("Error: cannot load in the image file '" + file.getName() + "'");
				System.exit(1);
			}
		}
		Example[] data = examples.toArray(new Example[examples.size()]);
		return new ExampleSet(data);
	}

}



class Input extends Layer {

	public Input(int w, int h, int d) {

		this.inW(w);
		this.inH(h);
		this.inD(d);

		this.outW(w);
		this.outH(h);
		this.outD(d);

		this.type = LayerType.input;
	}


	public Volume forward(double[] x) {
		Volume v = new Volume(this.inW(), this.inH(), this.inD(), x);
		return this.forward(v);
	}


	public Volume forward(Volume x) {
		this.input = x;
		this.output = x;
		return x;
	}


	@Override
	public void connect(Layer l) {


	}

}


class Convolution extends Layer {

	private Volume[] _filters;
	private int _stride;
	private int _pad;

	private int _filterW;
	private int _filterH;
	private int _filterD;


	public Convolution(int filterW, int filterH, int filterD, int stride, int pad, double bias) {
		this.type = LayerType.convolution;
		_filterW = filterW;
		_filterH = (filterH == 0) ? filterW : filterH;
		_filterD = filterD;
		_stride = (stride == 0) ? 1 : stride;
		_pad = pad;
		this.bias = bias;
	}


	public void connect(Layer l) {

		this.inW(l.outW());
		this.inH(l.outH());
		this.inD(l.outD());

		int w = (int) Math.floor((double) (this.inW() + _pad * 2 - _filterW) / _stride + 1);
		this.outW(w);

		int h = (int) Math.floor((double) (this.inH() + _pad * 2 - _filterH) / _stride + 1);
		this.outH(h);

		this.outD(_filterD);

		_filters = new Volume[_filterD];
		for (int i = 0; i < _filters.length; i++) {
			_filters[i] = new Volume(_filterW, _filterH, this.inD());
		}

		this.biases = new Volume(1, 1, this.outD(), this.bias);
	}


	public Volume forward(Volume c) {

		this.input = c;
		Volume A = new Volume(this.outW(), this.outH(), this.outD(), 0.0);

		int inW = c.width();
		int inH = c.height();

		for (int d = 0; d < this.outD(); d++) {
			Volume f = this._filters[d];
			int x = -this._pad;
			int y = -this._pad;
			for (int ay = 0; ay < this.outH(); y += _stride, ay++) {
				x = -this._pad;
				for (int ax = 0; ax < this.outW(); x += _stride, ax++) {
					double a = 0.0;
					for (int fy = 0; fy < f.height(); fy++) {
						int oy = y + fy;
						for (int fx = 0; fx < f.height(); fx++) {
							int ox = x + fx;
							if (oy >= 0 && oy < inH && ox >= 0 && ox < inW) {
								for (int fd = 0; fd < f.depth(); fd++) {
									a += f.W[((f.width() * fy) + fx) * f.depth() + fd] * c.W[((inW * oy) + ox) * c.depth() + fd];
								}
							}
						}
					}
					a += this.biases.W[d];
					A.set(ax, ay, d, a);
				}
			}
		}
		this.output = A;
		return this.output;
	}


	public void backward() {
		Volume in = this.input;
		in.dW = new double[in.W.length];

		int inW = in.width();
		int inH = in.height();

		for (int d = 0; d < this.outD(); d++) {
			Volume f = this._filters[d];
			int x = -this._pad;
			int y = -this._pad;
			for (int ay = 0; ay < this.outW(); y += _stride, ay++) {
				x = -this._pad;
				for (int ax = 0; ax < this.outW(); x += _stride, ax++) {
					double grad = this.output.getGrad(ax, ay, d);
					for (int fy = 0; fy < f.height(); fy++) {
						int oy = y + fy;
						for (int fx = 0; fx < f.width(); fx++) {
							int ox = x + fx;
							if (oy >= 0 && oy < inH && ox >= 0 && ox < inW) {
								for (int fd = 0; fd < f.depth(); fd++) {
									int ix1 = ((inW * oy) + ox) * in.depth() + fd;
									int ix2 = ((f.width() * fy) + fx) * f.depth() + fd;
									f.dW[ix2] += in.W[ix1] * grad;
									in.dW[ix1] += f.W[ix2] * grad;
								}
							}
						}
					}
					this.biases.dW[d] += grad;
				}
			}
		}
	}


	public Volume[] response() {
		Volume[] ret = new Volume[_filters.length + 1];
		for (int i = 0; i < _filters.length; i++) {
			ret[i] = _filters[i];
		}
		ret[_filters.length] = this.biases;
		return ret;
	}

}


class Pool extends Layer {

	private int _w;
	private int _h;
	private int _d;
	private int _stride;
	private int _pad;

	private int[] _mapx;
	private int[] _mapy;


	public Pool(int filterW, int filterH, int stride, int pad) {
		this.type = LayerType.pool;
		_w = filterW;
		_h = (filterH == 0) ? _w : filterH;
		_stride = (stride == 0) ? 1 : stride;
		_pad = pad;
	}


	public void connect(Layer l) {

		this.inW(l.outW());
		this.inH(l.outH());
		this.inD(l.outD());

		this.outD(this.inD());

		int w = (int) Math.floor((double) (this.inW() + _pad * 2 - _w) / _stride + 1);
		this.outW(w);

		int h = (int) Math.floor((double) (this.inH() + _pad * 2 - _h) / _stride + 1);
		this.outH(h);

		_mapx = new int[this.outLength()];
		_mapy = new int[this.outLength()];

	}


	public Volume forward(Volume c) {

		this.input = c;
		Volume A = new Volume(this.outW(), this.outH(), this.outD(), 0.0);

		int n = 0; // a counter for switches
		for (int d = 0; d < this.outD(); d++) {
			int x = -this._pad;
			int y = -this._pad;
			for (int ax = 0; ax < this.outW(); x += this._stride, ax++) {
				y = -this._pad;
				for (int ay = 0; ay < this.outH(); y += this._stride, ay++) {
					double a = -999999999;
					int px = -1;
					int py = -1;
					for (int fx = 0; fx < _w; fx++) {
						for (int fy = 0; fy < _h; fy++) {
							int oy = y + fy;
							int ox = x + fx;
							if (oy >= 0 && oy < c.height() && ox >= 0 && ox < c.width()) {
								double v = c.get(ox, oy, d);
								if (v > a) {
									a = v;
									px = ox;
									py = oy;
								}
							}
						}
					}
					this._mapx[n] = px;
					this._mapy[n] = py;
					n++;
					A.set(ax, ay, d, a);
				}
			}
		}

		this.output = A;
		return this.output;
	}


	public void backward() {
		Volume V = this.input;
		V.dW = new double[V.W.length];
		Volume A = this.output;
		int n = 0;
		for (int d = 0; d < this.outD(); d++) {
			int x = -this._pad;
			int y = -this._pad;
			for (int ax = 0; ax < this.outW(); x += this._stride, ax++) {
				y = -this._pad;
				for (int ay = 0; ay < this.outW(); y += this._stride, ay++) {
					double grad = this.output.getGrad(ax, ay, d);
					try {
						V.addGrad(_mapx[n], _mapy[n], d, grad);

						n++;
					}
					catch (Exception ex) {
						Lab3.writeLine(ex.getMessage());
					}
				}
			}
		}
	}

}




class FullConnect extends Layer {

	private Volume[] _units;


	public FullConnect(int units, double bias) {
		this.type = LayerType.fullconnect;
		_units = new Volume[units];
		this.bias = bias;
	}


	public void connect(Layer l) {

		this.inW(l.outW());
		this.inH(l.outH());
		this.inD(l.outD());

		this.outH(1);
		this.outW(1);
		this.outD(_units.length);

		for (int i = 0; i < _units.length; i++) {
			_units[i] = new Volume(1, 1, this.inLength());
		}

		this.biases = new Volume(1, 1, this.outD(), this.bias);
	}


	public Volume forward(Volume x) {
		this.input = x;

		Volume out = new Volume(1, 1, this.outD(), 0);
		this.output = out;

		for (int i = 0; i < _units.length; i++) {
			out.W[i] = x.dot(_units[i].W) + this.biases.W[i];
		}

		return out;
	}


	public void backward() {
		double grad;
		Volume in = this.input;
		in.dW = new double[in.W.length];

		for (int i = 0; i < _units.length; i++) {
			Volume unit_i = _units[i];
			grad = this.output.dW[i];
			for (int d = 0; d < this.inLength(); d++) {
				in.dW[d] += unit_i.W[d] * grad;
				unit_i.dW[d] += in.W[d] * grad;
			}
			this.biases.dW[i] += grad;
		}
	}


	public Volume[] response() {
		Volume[] ret = new Volume[_units.length + 1];
		for (int i = 0; i < _units.length; i++) {
			ret[i] = _units[i];
		}
		ret[_units.length] = this.biases;
		return ret;
	}

}




class ActivationLayer extends Layer {

	public ActivationLayer() {

	}


	public void connect(Layer l) {

		this.inW(l.outW());
		this.inH(l.outH());
		this.inD(l.outD());
		this.outW(this.inW());
		this.outH(this.inH());
		this.outD(this.inD());
	}


	public Volume forward(Volume x) {
		this.input = x;
		Volume out = new Volume(x, 0);
		this.output = out;
		double[] f = x.W;
		double[] g = out.W;
		this.computeForward(f, g);
		return out;
	}


	public void backward() {
		Volume in = this.input;
		Volume out = this.output;
		in.dW = new double[in.W.length];
		double[] w = out.W;
		double[] dw = in.dW;
		this.computeBackward(w, dw);
	}


	protected void computeForward(double[] in, double[] out) {

	}


	protected void computeBackward(double[] w, double[] dw) {

	}

}

class LeRu extends ActivationLayer {

	public LeRu() {
		this.type = LayerType.leru;
	}


	private double noise() {
		return Math.random() * 0.001;
	}

	public Volume forward(Volume x) {
		this.input = x;
		Volume out = new Volume(x);
		int n = x.W.length;
		double[] outW = out.W;
		for (int i = 0; i < n; i++) {
			if (outW[i] < 0) outW[i] = 0;// = noise();
		}
		this.output = out;
		return out;
	}


	public void backward() {
		Volume in = this.input;
		Volume out = this.output;
		int n = in.W.length;
		in.dW = new double[n];
		for (int i = 0; i < n; i++) {
			if (out.W[i] <= 0) in.dW[i] = 0; //= noise();
			else in.dW[i] = out.dW[i];
		}
	}

}



class LossLayer extends Layer {

	public void connect(Layer l) {

		this.inW(l.outW());
		this.inH(l.outH());
		this.inD(l.outD());

		this.outW(1);
		this.outH(1);
		this.outD(this.inLength());
	}

}


class Softmax extends LossLayer {

	private double[] _prob;


	public Softmax() {
		this.type = LayerType.softmax;
	}


	public Volume forward(Volume v) {
		this.input = v;
		Volume out = new Volume(1, 1, this.outD(), 0.0);

		double[] as = v.W;
		double amax = v.W[0];

		for (int i = 1; i < this.outD(); i++) {
			if (as[i] > amax) amax = as[i];
		}

		double[] prob = new double[this.outD()];
		double sum = 0.0;
		for (int i = 0; i < this.outD(); i++) {
			double e = Math.exp(as[i] - amax);
			sum += e;
			prob[i] = e;
		}


		for (int i = 0; i < this.outD(); i++) {
			prob[i] /= sum;
			out.W[i] = prob[i];
		}

		this._prob = prob;
		this.output = out;
		return this.output;

	}


	public double backward(double[] v) {
		double max = v[0];
		int y = 0;
		for (int i = 1; i < v.length; i++) {
			if (v[i] > max) {
				y = i;
				max = v[i];
			}
		}

		Volume x = this.input;
		x.dW = new double[x.W.length];

		for (int i = 0; i < this.outD(); i++) {
			double indicator = (i == y) ? 1.0 : 0.0;
			double c = -(indicator - this._prob[i]);
			x.dW[i] = c;
		}

		return -Math.log(this._prob[y]);
	}

}


class SGDTrainer extends Trainer {

	private double _rate;
	private double _momentum;
	private double _decayL1;
	private double _decayL2;
	private double _decayLossL1;
	private double _decayLossL2;
	private double _loss = 0.0;
	private int _batchSize;



	public SGDTrainer(double learningRate, int batchSize, double momentum, double decayL1, double decayL2) {
		_rate = learningRate;
		_momentum = momentum;
		_decayL1 = decayL1;
		_decayL2 = decayL2;
		_batchSize = batchSize;
	}


	@Override
	protected void trainOneExample(ConvNet net, double[] x, double[] y) {

		_decayLossL1 = 0.0;
		_decayLossL2 = 0.0;
		_loss = 0.0;


		//timer.start();
		this.net().forward(x);
		//timer.stop();
		//_forwardTime = timer.elapsedTime();

		//timer.start();
		_loss = this.net().backward(y);
		//		timer.stop();
		//		_backwardtime = timer.elapsedTime();

		if ((this.step() % _batchSize) == 0) {

			Volume[] r = this.net().response();

			double[][] gs = new double[r.length][];
			for (int i = 0; i < r.length; i++) {
				gs[i] = new double[r[i].dW.length];
			}

			for (int i = 0; i < r.length; i++) {

				double[] w = r[i].W;
				double[] g = r[i].dW;

				for (int j = 0; j < w.length; j++) {

					_decayLossL1 += _decayL1 * Math.abs(w[j]);
					_decayLossL2 += _decayL2 * w[j] * w[j] / 2;

					double gradL1 = _decayL1 * (w[j] > 0 ? 1 : -1);
					double gradL2 = _decayL2 * (w[j]);

					double delta = (gradL1 + gradL2 + g[j]) / _batchSize;
					double[] gs_i = gs[i];

					if (_momentum > 0.0) {
						double dx = _momentum * gs_i[j] - _rate * delta;
						gs_i[j] = dx;
						w[j] += dx;
					}
					else {
						w[j] += -_rate * delta;
					}
					g[j] = 0.0;
				}

			}

		}

	}

}

class DropOut extends Layer {

	private boolean[] _dropped;

	private double _dropProb;


	public DropOut(double dropOutProb) {
		this.type = LayerType.dropout;
		_dropProb = dropOutProb;
	}


	public void connect(Layer l) {

		this.inW(l.outW());
		this.inH(l.outH());
		this.inD(l.outD());

		this.outW(this.inW());
		this.outH(this.inH());
		this.outD(this.inD());
		_dropped = new boolean[this.outLength()];
	}


	public Volume forward(Volume x) {
		this.input = x;

		Volume out = new Volume(x);
		int m = x.W.length;

		if (this.training()) {
			for (int i = 0; i < m; i++) {
				if (Math.random() < this._dropProb) {
					out.W[i] = 0;
					this._dropped[i] = true;
				} // drop!
				else {
					this._dropped[i] = false;
				}
			}
		}
		else {
			for (int i = 0; i < m; i++) {
				out.W[i] *= this._dropProb;
			}
		}

		this.output = out;
		return output;
	}


	public void backward() {
		Volume in = this.input;
		Volume grad = this.output;

		in.dW = new double[in.W.length];
		for (int i = 0; i < in.W.length; i++) {
			if (!(this._dropped[i])) {
				in.dW[i] = grad.dW[i];
			}
		}
	}

}


public class Lab3 {

	private static String[] _cats = new String[] { "airplane", "butterfly", "flower", "piano", "starfish", "watch" };


	public static String sprintf(String format, Object... values) {
		return String.format(format, values);
	}

	public static void writeLine(String msg) {
		System.out.println(msg);
	}

	private static ExampleSet[] loadImageDataSets(String trainDir, String tuneDir, String testDir, int imageSize, ImageUtil.LoadOption option) {
		ImageDataSetReader train = new ImageDataSetReader(trainDir, _cats, imageSize, option);
		ImageDataSetReader tune = new ImageDataSetReader(tuneDir, _cats, imageSize, option);
		ImageDataSetReader test = new ImageDataSetReader(testDir, _cats, imageSize, option);
		ExampleSet[] set = new ExampleSet[3];
		set[0] = train.readDataSet();
		set[1] = tune.readDataSet();
		set[2] = test.readDataSet();
		return set;
	}


	public static double[] maxOutVector(double[] v) {
		double[] out = new double[v.length];
		double max = v[0];
		int y = 0;
		for (int i = 1; i < v.length; i++) {
			if (v[i] > max) {
				y = i;
				max = v[i];
			}
		}
		out[y] = 1;
		return out;
	}


	private static int maxOut(double[] y) {
		int maxi = 0;
		double max = y[0];
		for (int i = 1; i < y.length; i++) {
			if (y[i] > max) {
				max = y[i];
				maxi = i;
			}
		}
		return maxi;
	}


	private static double computeError(ConvNet net, Example[] test) {
		int err = 0;
		for (Example e : test) {
			double[] yhat = net.predict(e.x.W);
			int p = maxOut(yhat);
			int a = maxOut(e.y.W);
			if (p != a) {
				err++;
			}
		}
		double rate = (double) err / test.length;
		return rate;
	}


	private static void saveErrorImages(ConvNet net, Example[] test) {
		int cnt = 0;
		for (Example e : test) {
			double[] yhat = net.predict(e.x.W);
			int p = maxOut(yhat);
			int a = maxOut(e.y.W);
			if (p != a) {
				cnt++;
				ImageUtil.saveImage(e.x, "./bin/images/" + (cnt + "_" + a + "_" + p) + ".png");
			}
		}
	}




	private static double printConfusionMatrix(ConvNet net, Example[] test) {
		int w = test[0].y.W.length;
		int[][] confusion = new int[w][w];
		int err = 0;

		for (Example e : test) {
			double[] yhat = net.predict(e.x.W);
			int p = maxOut(yhat);
			int a = maxOut(e.y.W);
			confusion[p][a]++;
			if (p != a) err++;
		}

		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < _cats.length; i++) {
			sb.append(sprintf("%10s", _cats[i]));
		}
		writeLine("          " + sb.toString());

		for (int i = 0; i < confusion.length; i++) {
			sb = new StringBuffer();
			for (int j = 0; j < confusion[i].length; j++) {
				sb.append(sprintf("%10d", confusion[i][j]));
			}
			writeLine(sprintf("%10s", _cats[i]) + sb.toString());
		}

		return ((double) err) / test.length;
	}


	private static Example[] sampleExamples(Example[] data, double frac) {
		List<Example> items = new ArrayList<Example>();
		int n = (int) (frac * data.length);
		for (Example e : data) {
			items.add(e);
		}
		Example[] ret = new Example[n];
		for (int i = 0; i < n; i++) {
			ret[i] = items.get(i);
		}
		return ret;
	}



	private static void learningCurve(ExampleSet[] dataSets) {

		Example[] train = dataSets[0].examples();
		Example[] tune = dataSets[1].examples();
		Example[] test = dataSets[2].examples();

		for (int i = 10; i <= 100; i = i + 10) {

			train = sampleExamples(train, ((double) i) / 100);

			ConvNet net = new ConvNet();

			Example ex = train[0];

			net.addLayer(new Input(ex.x.width(), ex.x.height(), ex.x.depth()));

			net.addLayer(new Convolution(5, 5, 25, 1, 2, 1.0));
			net.addLayer(new LeRu());
			net.addLayer(new Pool(2, 2, 2, 1));

			net.addLayer(new Convolution(5, 5, 16, 1, 2, 1.0));
			net.addLayer(new LeRu());
			net.addLayer(new Pool(2, 2, 2, 1));

			net.addLayer(new Convolution(5, 5, 20, 1, 2, 1.0));
			net.addLayer(new LeRu());
			net.addLayer(new Pool(2, 2, 2, 1));
			net.addLayer(new DropOut(0.5));

			net.addLayer(new FullConnect(ex.y.depth(), 1.0));
			net.addLayer(new Softmax());

			double eta = 0.007;
			double alpha = 0.90;
			double lambda = 0.0001;

			Trainer trainer = new SGDTrainer(eta, 4, alpha, 0.005, lambda);

			net.epochs = 50;

			trainer.train(net, train, tune);

			writeLine("Tune examples: " + tune.length);
			double err = printConfusionMatrix(net, tune);
			writeLine("Test set accuracy: " + sprintf("%1.8f", (1 - err)));

			writeLine("Tune examples: " + test.length);
			err = printConfusionMatrix(net, test);
			writeLine("Test set accuracy: " + sprintf("%1.8f", (1 - err)));
			writeLine("");

		}
	}


	public static void trainAndTest(ExampleSet[] dataSets, int epochs) {

		ConvNet net = new ConvNet();
		Example ex = dataSets[0].get(0);

		net.addLayer(new Input(ex.x.width(), ex.x.height(), ex.x.depth()));

		net.addLayer(new Convolution(5, 5, 25, 1, 2, 1.0));
		net.addLayer(new LeRu());
		net.addLayer(new Pool(2, 2, 2, 1));

		net.addLayer(new Convolution(5, 5, 20, 1, 2, 1.0));
		net.addLayer(new LeRu());
		net.addLayer(new Pool(2, 2, 2, 1));

		net.addLayer(new Convolution(5, 5, 16, 1, 2, 1.0));
		net.addLayer(new LeRu());
		net.addLayer(new Pool(2, 2, 2, 1));
		
		net.addLayer(new FullConnect(ex.y.depth(), 1.0));
		net.addLayer(new Softmax());

		double eta = 0.001;
		double alpha = 0.90;
		double lambda = 0.0001;

		Trainer trainer = new SGDTrainer(eta, 4, alpha, 0.005, lambda);

		trainer.onEpoch(t -> {
			writeLine("Epoch: " + t.epoch());
			double trainerr;
			double testerr;
			double tuneerr;

			//writeLine("Train size: " + dataSets[0].examples().length);
			trainerr = printConfusionMatrix(net, dataSets[0].examples());
			writeLine("Train accuracy: " + sprintf("%1.8f", (1 - trainerr)));
			writeLine("");

			//writeLine("Tune size: " + dataSets[1].examples().length);
			tuneerr = printConfusionMatrix(net, dataSets[1].examples());
			writeLine("Tune accuracy: " + sprintf("%1.8f", (1 - tuneerr)));
			writeLine("");

			//writeLine("Test size: " + dataSets[2].examples().length);
			testerr = printConfusionMatrix(net, dataSets[2].examples());
			writeLine("Test accuracy: " + sprintf("%1.8f", (1 - testerr)));
			writeLine("");
			writeLine("");

			if (trainerr < 0.02 || tuneerr < 0.21 || testerr <0.21) return false;

			return true;
		});


		net.epochs = epochs;
		trainer.train(net, dataSets[0].examples(), dataSets[1].examples());

		writeLine("** Final Results **");
		double err = printConfusionMatrix(net, dataSets[2].examples());
		writeLine("Test accuracy: " + sprintf("%1.8f", (1 - err)));
		writeLine("");
		writeLine("");

		writeLine("");
	}


	public static void main(String[] args) {

		String trainDirectory = "images/trainset/";
		String tuneDirectory = "images/tuneset/";
		String testDirectory = "images/testset/";

		int imageSize = 32;

		if (args.length > 5 || (args.length < 3 && args.length!=0)) {
			System.err.println("Usage error: java Lab3 <train_set_folder_path> <tune_set_folder_path> <test_set_folder_path> <imageSize>");
			System.exit(1);
		}
		if (args.length >= 1) {
			trainDirectory = args[0];
		}
		if (args.length >= 2) {
			tuneDirectory = args[1];
		}
		if (args.length >= 3) {
			testDirectory = args[2];
		}
		if (args.length >= 4) {
			imageSize = Integer.parseInt(args[3]);
		}


		ExampleSet[] dataSets = loadImageDataSets(trainDirectory, tuneDirectory, testDirectory, imageSize, ImageUtil.LoadOption.RGB_EDGES);

		trainAndTest(dataSets, 150);
	}

}
