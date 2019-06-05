    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(WaterBottleDataset, self).load_mask(image_id)








# Train or validation dataset?
        assert subset in ["train", "val"]
        #dataset_dir = os.path.join(dataset_dir, subset)

        # Add classes. We have three classes to add.
        self.add_class("water_bottle", 1, "cap")
        self.add_class("water_bottle", 2, "bottle")
        self.add_class("water_bottle", 3, "label")


        #annotations = json.load(open(os.path.join(dataset_dir, "instances_{}.json".format(subset))))   # Could make the following for loop use "image_id" in annotations instead



        image_ids = list(range(1,251))
        # Add images
        for image_id in image_ids:
            self.add_image(
                "water_bottle",
                image_id= image_id,  # use file name as a unique image id
                path= os.path.join(dataset_dir, "bottle_{}/{:04d}.png".format(subset,image_id)))
