import argparse

def parse_opt():
	parser = argparse.ArgumentParser()

	parser.add_argument("-tf", "--trainFolder", type=str, 
		default=r"E:\dataset\detection\pubtables\test",
		help='Train Folder path')

	parser.add_argument("-vf", "--valFolder", type=str, 
		# default=None,
		default=r"E:\dataset\detection\pubtables\val",
		help='Val Folder path')

	parser.add_argument("-imgsz", "--imageSize", type=int, default=416, help="Image size")

	parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of Epochs")

	parser.add_argument("-cpfld", "--checkpointFolder", type=str, default="saved_models", help="Where to store model output")

	parser.add_argument("-cpfle", "--checkpointFile", type=str, default="model", help="Model name")

	parser.add_argument("-bs", "--batchSize", type=int, default=32, help="Batch size for model training")

	parser.add_argument("-nc", "--numClasses", type=int, 
		# default=80, 
		default=6, 
		help="")

	parser.add_argument("-", "--patience", type=int, default=5, help="")

	parser.add_argument("-optim", "--optimizer", type=str, default="sgd", help="Optimzer for training")

	# parser.add_argument("-", "--", type=str, default="", help="")

	parser.add_argument("-lr", "--lr", type=float, default=0.0001, help="")
	parser.add_argument("-lrf", "--lrf", type=float, default=0.0001, help="")
	
	# parser.add_argument("-", "--", type=str, default="", help="")
	args = vars(parser.parse_args())
	return args



	