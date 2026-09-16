# Adapters and session

Geometry presentation can be tagged per instance with `SceneSource.geom_role` (`GeometryRole`
bits), `geom_group_visible` (default-view mask), and `geom_collision_mesh` (optional alternate
collision shape). Empty columns retain the behavior of older providers. These fields separate
display choices from collision-engine state; renderers consume the same neutral metadata.

## Adapter protocol

::: mojive.adapters.base.AdapterCaps

::: mojive.adapters.base.SceneSaveOptions

::: mojive.adapters.base.SceneProvider

::: mojive.adapters.base.SceneAdapterBase

::: mojive.adapters.base.SceneAdapter

## Capability interfaces

::: mojive.adapters.base.SceneInspection

::: mojive.adapters.base.SimulationControl

::: mojive.adapters.base.SceneDocuments

::: mojive.adapters.base.SceneEditing

::: mojive.adapters.base.KeyframeEditing

::: mojive.adapters.base.ModelEditing

## Focused consumer interfaces

::: mojive.adapters.base.SceneRuntime

::: mojive.adapters.base.KeyframeCatalog

::: mojive.adapters.base.KeyframePlayback

::: mojive.adapters.base.DocumentCheckpoint

::: mojive.adapters.base.PointPerturbation

::: mojive.adapters.base.PhysicsOption

::: mojive.adapters.base.PhysicsOptions

## Application session

::: mojive.session.Session

::: mojive.session.PerturbState

::: mojive.session.SceneOverrides

## Commands

::: mojive.commands

## Conformance

::: mojive.adapters.conformance

## Adapter registration

::: mojive.adapters.registry.register_adapter

::: mojive.adapters.registry.unregister_adapter
