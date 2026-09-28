from toolkit.basic import flush


class PhasedLoadMixin:
    """Phased loading contract shared by training-capable models.

    Each phase loads one resident component. Components place themselves, and
    model_config drives quantization/offload/placement per component role
    (see BaseModel.component_load_kwargs). The training process calls the
    three phases one at a time -- vae, text encoder, transformer -- running
    dataset preparation between phases, and may free the vae after the text
    encoder phase (see BaseSDTrainProcess.vae_training_mode). load_model()
    composes all three at once for inference/generation; it loads the big
    component first so progress reads well, which is fine because every
    component pulls the others it needs at use time.

    Model-side requirements:
        _load_transformer() -> transformer module
        _load_text_encoder() -> (tokenizer, text_encoder, extras) where extras
            is a dict of extra attributes (processors etc.) assigned to self
        _load_vae() -> vae module
        get_train_scheduler() -> the training noise scheduler
        _build_pipeline() -> a pipeline bound to this holder
        display_name: short model name used in status updates

    Optional hooks:
        _transformer_pre_post_load(transformer)
        _text_encoder_pre_post_load(text_encoder)
        _post_model_load() -- runs at the end of the all-at-once load
    """

    display_name = "model"

    def load_transformer(self):
        """Load the denoiser, then quantize/offload/place per model_config.

        This is the final load step of the phased training startup, so the
        holder wiring (noise scheduler + pipeline) is completed here. The
        pipeline stores only the holder reference, so building it before the
        other components (as load_model does) is equivalent."""
        transformer = self._load_transformer()
        self._transformer_pre_post_load(transformer)

        # quantize + offload + placement, all driven by model_config
        transformer.aitk_post_load(**self.component_load_kwargs("transformer"))
        self.model = transformer

        self.noise_scheduler = self.get_train_scheduler()
        self.pipeline = self._build_pipeline()
        flush()

    def _transformer_pre_post_load(self, transformer):
        pass

    def load_text_encoder(self):
        """Load the text encoder (with tokenizer and any processors), then
        quantize/offload/place per model_config."""
        tokenizer, text_encoder, extras = self._load_text_encoder()
        self._text_encoder_pre_post_load(text_encoder)

        # quantize + offload + placement, all driven by model_config
        text_encoder.aitk_post_load(**self.component_load_kwargs("te"))
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        for attr, value in extras.items():
            setattr(self, attr, value)
        flush()

    def _text_encoder_pre_post_load(self, text_encoder):
        pass

    def load_vae(self):
        """Load the vae and place it per model_config."""
        vae = self._load_vae()
        vae.to(self.vae_device_torch, dtype=self.vae_torch_dtype)
        self.vae = vae

    def _post_model_load(self):
        pass

    def load_model(self):
        self.print_and_status_update(f"Loading {self.display_name} model")
        # all-at-once load used by inference/generation. The training process
        # loads these steps one at a time instead, so that dataset prep can
        # run while only the small helper components are resident.
        self.load_transformer()
        self.load_text_encoder()
        self.load_vae()
        self._post_model_load()
        self.print_and_status_update("Model Loaded")
